import numpy as np
import torch
from typing import Tuple, List, Union

from utilsv2 import LagrangeEmbedding


class VisionNet(torch.nn.Module):
    def pre_proc(self, raw_data, **kwargs):
        pre_coeff = torch.tensor([  # 
            [[[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]]],
            [[[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]]], 
            [[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], 
            [[[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]], 
            ], dtype=torch.float32, device=raw_data.device) / 3 # [out_C, in_C, kernel_H, kernel_W]
        kh, kw, sh, sw = pre_coeff.shape[2], pre_coeff.shape[3], 2, 2
        data = raw_data.unfold(2, kh, sh).unfold(3, kw, sw)  # [N, in_C, out_H, out_W, kernel_H, kernel_W]
        data = torch.einsum("nihwpq,jipq->njhw", data, pre_coeff)  # [N, out_C, out_H, out_W]
        data = data.unfold(2, kh, sh).unfold(3, kw, sw)  # [N, out_C, out_H, out_W, kernel_H, kernel_W]
        data = torch.einsum("nkhwpq,jipq->nkjhw", data, pre_coeff)  # [N, out_C, out_C, out_H, out_W]
        data = data.reshape(-1, 1)  # [N * out_C * out_C * out_H * out_W, 1]
        return data
    
    def post_proc(self, x, *args):
        return x.reshape(-1, self.hidden_units)
    
    def __init__(self, data_loader: torch.utils.data.dataloader.DataLoader, dof: int, n_cls: int, gen_size=Tuple[int]):
        super(VisionNet, self).__init__()

        self.backbone = LagrangeEmbedding(data_loader, dof, pre_proc=self.pre_proc)

        self.dof, self.n_cls = dof, n_cls
        self.hidden_units = 576 * dof  # out_C ** 2 * out_H * out_W * dof
        self.cls_fc = torch.nn.Linear(self.hidden_units, n_cls, bias=False)
        self.gen_fc = torch.nn.Linear(self.hidden_units, gen_size[0] * gen_size[1], bias=False)

    def forward(self, x):
        x = self.pre_proc(x)  # [N, in_C, in_H, in_W] -> [N * out_C * out_C * out_H * out_W, 1]
        basis_values = self.backbone(x)  # [N * out_C ** 2 * out_H * out_W, dof]
        x = self.post_proc(basis_values)  # [N, out_C ** 2 * out_H * out_W * dof]
        return self.cls_fc(x), self.gen_fc(x)
    
    def compute_regularization(self):
        weights, = self.cls_fc.parameters()
        weights = weights.view(-1, self.dof, self.n_cls)
        return torch.mean(torch.square(weights[:, self.backbone.edges[:, 0], :] - weights[:, self.backbone.edges[:, 1], :]))

if __name__ == "__main__":
        device="cuda"

        import torchvision
        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt

        # Load and preprocess the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        def collate_batch(batch): return torch.stack([x for x, _ in batch]).to(device), torch.tensor([y for _, y in batch]).to(device)
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_batch)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_batch)

        n_cls, gen_size = 10, (32, 32)
        model = VisionNet(train_loader, dof=16, n_cls=n_cls, gen_size=gen_size).to(device)

        # --------------------------------------------------
        # Define loss function and optimizer
        cls_criterion = torch.nn.CrossEntropyLoss()
        cls_optimizer = torch.optim.Adam(model.cls_fc.parameters(), lr=0.001)

        # Training loop
        for epoch in range(10):  # Adjust the number of epochs as needed
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                cls_optimizer.zero_grad()
                cls_outputs, _ = model(inputs)
                loss = cls_criterion(cls_outputs, labels)  # + model.compute_regularization()
                loss.backward()
                cls_optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"\r[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}", end="")
                    running_loss = 0.0

            # Calculate test accuracy
            model.eval()  # Set the model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
                for (images, labels) in test_loader:
                    cls_outputs, _ = model(images)
                    _, predicted = torch.max(cls_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f"\tTest Accuracy: {100 * correct / total:.2f}%")
        print("Finished Training")

        # --------------------------------------------------
        # Define loss function and optimizer
        gen_criterion = torch.nn.MSELoss()
        gen_optimizer = torch.optim.Adam(model.gen_fc.parameters(), lr=0.0001)  # [5,   900] loss: 0.004  Test MSE: 0.0003

        # Training loop
        for epoch in range(5):  # Adjust the number of epochs as needed
            running_loss = 0.0
            for i, (inputs, _) in enumerate(train_loader, 0):
                labels = torch.nn.functional.interpolate(inputs, size=gen_size, mode='bicubic')
                
                gen_optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = gen_criterion(outputs.view(-1, gen_size[0] * gen_size[1]), labels.view(-1, gen_size[0] * gen_size[1]))
                loss.backward()
                gen_optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"\r[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}", end="")
                    running_loss = 0.0

            # Calculate test accuracy
            model.eval()  # Set the model to evaluation mode
            mse = 0
            total = 0
            with torch.no_grad():
                for (images, _) in test_loader:
                    # images = images.to(device)
                    labels = torch.nn.functional.interpolate(images, size=gen_size)

                    _, outputs = model(images)
                    loss = gen_criterion(outputs.view(-1, gen_size[0] * gen_size[1]), labels.view(-1, gen_size[0] * gen_size[1]))
                    total += labels.size(0)
                    mse += loss
            print(f"\tTest MSE: {mse / total:.4f}")
        print("Finished Training")

        org_img = images[0, 0].to("cpu").detach().numpy().reshape(28, 28)
        tar_img = torch.nn.functional.interpolate(images, size=(gen_size[0], gen_size[1]), mode='bicubic')[0, 0].to("cpu").detach().numpy()
        pred_img = outputs[0].to("cpu").detach().numpy().reshape(gen_size[0], gen_size[1])

        plt.subplot(1, 3, 1)
        plt.imshow(org_img, cmap='viridis', interpolation='none')
        plt.title('original image')

        plt.subplot(1, 3, 2)
        plt.imshow(tar_img, cmap='viridis', interpolation='none')
        plt.title('target image')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_img, cmap='viridis', interpolation='none')
        plt.title('prediction image')

        plt.show()
