from typing import Tuple, List, Union, Optional, Callable
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utilsv2 import LagrangeEmbedding


torch.manual_seed(0)


# GPU: RTX 2080Ti
# +--------------+---------------------+----------+----------+-----------+
# | dataset      | apply LagrangeEmbed | data aug | test acc | init time |
# +==============+=====================+==========+==========+===========+
# | minist       | False               | False    | 99.14%   |           |
# | minist       | True                | False    | 99.17%   | 1.55s     |
# +--------------+---------------------+----------+----------+-----------+
# | fashionmnist | False               | False    | 92.47%   |           |
# | fashionmnist | True                | False    | 92.56%   | 1.55s     |
# +--------------+---------------------+----------+----------+-----------+
# | fashionmnist | False               | True     | 92.14%   |           |
# | fashionmnist | True                | True     | 92.14%   | 2.08s     |
# +--------------+---------------------+----------+----------+-----------+
# | cifar10      | False               | False    | 71.43%   |           |
# | cifar10      | True                | False    | 71.54%   | 1.83s     |
# +--------------+---------------------+----------+----------+-----------+
# | cifar10      | False               | True     | 71.93%   |           |
# | cifar10      | True                | True     | 72.09%   | 2.29s     |
# +--------------+---------------------+----------+----------+-----------+
# | cifar100     | False               | False    | 36.50%   |           |
# | cifar100     | True                | False    | 36.78%   | 1.84s     |
# +--------------+---------------------+----------+----------+-----------+
# | cifar100     | False               | True     | 36.58%   |           |
# | cifar100     | True                | True     | 36.97%   | 2.30s     |
# +--------------+---------------------+----------+----------+-----------+


class Net(nn.Module):
    def __init__(self, in_channels, out_features, n_cls):
        super(Net, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25), nn.Flatten(1), 
            nn.Linear(out_features, 128), nn.ReLU(), nn.Dropout(0.5), 
            )
        self.branch = nn.Linear(128, n_cls)

    def forward(self, x):
        x = self.projection(x)
        x = self.branch(x)
        return x


class SimpleLagrangeEmbedding(LagrangeEmbedding):
    def build_mesh_from_dataloader(self, data_loader: torch.utils.data.dataloader.DataLoader, dof: int, pre_proc: Optional[Callable] = None, min_x: Optional[torch.Tensor] = None, max_x: Optional[torch.Tensor] = None):
        """Please set the first element in each batch as the `raw_data`. e.g., `images` is the `raw_data`, then `for images, labels in data_loader: ...`."""
        start_time = time.time()
        print("Initializing the input domain")
        min_x, max_x = None, None
        for batch in data_loader:
            batch_x = pre_proc(batch[0])
            if min_x is None or max_x is None:
                min_x = torch.min(batch_x, axis=0)[0]
                max_x = torch.max(batch_x, axis=0)[0]
            else:
                min_x = torch.minimum(torch.min(batch_x, axis=0)[0], min_x)
                max_x = torch.maximum(torch.max(batch_x, axis=0)[0], max_x)
        self.min_x, self.max_x, self.n_features, self.dof = min_x, max_x, torch.numel(min_x), dof
        self._points = torch.linspace(0., 1.0, dof, dtype=min_x.dtype, device=min_x.device)[:, None]
        self._simplices = torch.stack([torch.arange(1, dof, dtype=torch.int64, device=min_x.device), torch.arange(dof - 1, dtype=torch.int64, device=min_x.device)], dim=1)
        print("\nTime of Mesh Refinement: {:.2f}s".format(time.time() - start_time))
    
    def __init__(self, data_loader: torch.utils.data.dataloader.DataLoader, dof:int, pre_proc: Callable):
        """Compute first-order elements, then create basis functions."""
        self.min_x, self.max_x, self.n_features, self.dof = None, None, None, None
        self._points, self._simplices = None, None

        super(SimpleLagrangeEmbedding, self).__init__(data_loader, dof, pre_proc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, dim] -> [N, p]
        if self.inv.device != x.device:
            self.inv = self.inv.to(x.device)
        if self.p2t_mask.device != x.device:
            self.p2t_mask = self.p2t_mask.to(x.device)
        x = (x - self.min_x[None, :]) / (self.max_x[None, :] - self.min_x[None, :])
        x = x.reshape(-1, 1)
        
        # Convert point coordinates (x, y) to barycentric coordinates (u, v, w). 
        # A point is considered inside the simplex if and only if all the barycentric coordinates are non-negative. 
        # [1] The tensor `x` containing point coordinates. It has shape [b, dim], where b is the number of points and dim is the number of dimensions.
        # [2] The tensor `barycentric_values` containing barycentric coordinates of points. It has shape [N, t, dim+1], where t is the number of simplices.
        barycentric_values = torch.einsum("tdv,bd->btv", self.inv[:, :-1, :], x) + self.inv[None, :, -1, :] # [N, t, dim+1]

        # The tensor `n2t_mask` indicates the relationship between input points and simplices.  It has shape [N, t].
        # If n2t_mask[i, j] is True, it signifies that the i-th point is located inside the j-th simplex.
        n2t_mask = torch.all(barycentric_values >= 0, dim=-1).to(torch.float32)  # [N, t]

        # Gather the matching tensors p2t_mask and n2t_mask
        mask = torch.einsum("bt,ptv->bp", n2t_mask, self.p2t_mask) # [N, p]
        
        # Calculate the function values on all shape functions of the current basis function.
        # shape_values[i,j] signifies that the value of j-th basis function in i-th point.
        # print(n2t_mask.shape, self.p2t_mask.shape, barycentric_values.shape)
        # shape_values = torch.einsum("bt,ptv,btv->bp", n2t_mask, self.p2t_mask, barycentric_values) # [N, p]
        shape_values = torch.einsum("ptv,btv->bpt", self.p2t_mask, barycentric_values) # [N, p]
        shape_values = torch.einsum("bt,bpt->bp", n2t_mask, shape_values) # [N, p]
        
        # Concatenate the shape functions corresponding to the basis function 
        # and update the values of the basis function within the support boundary of all shape functions.
        basis_values = shape_values / torch.maximum(mask, torch.ones_like(mask))

        x = basis_values.reshape(-1, self.n_features, self.dof)
        x = x * (self.max_x - self.min_x)[None, :, None] + self.min_x[None, :, None]
        x = x.reshape(-1, self.n_features * self.dof)
        return x.detach()


class LagrangeNet(torch.nn.Module):
    def pre_proc(self, raw_data, **kwargs):
        data = self.frozen_pre_proc(raw_data).detach()  # Frozen the ToyCNN's projecjtion, the result will never have forward mode AD gradients!
        if self._ratio is None:
            self._ratio = torch.numel(data[0])
        return data.reshape(-1, 1)
    
    def post_proc(self, x, *args):
        return x.reshape(-1, self.hidden_units)
    
    def __init__(self, data_loader: torch.utils.data.dataloader.DataLoader, frozen_pre_proc, dof: int, n_cls: int):
        super(LagrangeNet, self).__init__()
        self._ratio, self.frozen_pre_proc = None, frozen_pre_proc
        self.dof = dof

        self.backbone = SimpleLagrangeEmbedding(data_loader, dof, pre_proc=self.pre_proc)  # The `LagrangeEmbedding` is train-free!
        self.hidden_units = self._ratio * dof
        self.branch = torch.nn.Linear(self.hidden_units, n_cls, bias=False)

    def forward(self, x):
        x = self.pre_proc(x)
        basis_values = self.backbone(x)
        x = self.post_proc(basis_values)
        return self.branch(x)


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    model.train()
    
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_freq == print_freq - 1:
            avg_loss = running_loss / print_freq
            print(f"\r[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}", end="")
            running_loss = 0.0

def test(test_loader, model, criterion):
    model.eval()

    runing_loss = 0.0
    runing_acc = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            runing_acc += (predicted == labels).sum().item()
            runing_loss += criterion(outputs, labels).item() * labels.size(0) # + model.compute_regularization()
    print(f"\tTest Loss: {runing_loss / total:.3f} Test Accuracy: {100 * runing_acc / total:.2f}%")


def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch LagrangeEmbedding Test')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--no-da', action='store_true', default=False, help='disables data augmentation')
    parser.add_argument('--lr', default=1.0, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', default=14, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency')
    parser.add_argument('--dof', default=2, type=int, metavar='N', help='degrees of freedom')
    args = parser.parse_args()
    device="cuda"
    print(args, args.no_da)

    # Diverse initialization
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == "mnist":
        print("Loading MNIST dataset...")
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        in_features, out_features, n_cls = 1, 9216, 10
    elif args.dataset == "fashionmnist":
        print("Loading FashionMNIST dataset...")
        if not args.no_da:
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
        in_features, out_features, n_cls = 1, 9216, 10
    elif args.dataset == "cifar10":
        print("Loading CIFAR10 dataset...")
        if not args.no_da:
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        in_features, out_features, n_cls = 3, 12544, 10
    elif args.dataset == "cifar100":
        print("Loading CIFAR100 dataset...")
        if not args.no_da:
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        in_features, out_features, n_cls = 3, 12544, 100
    else:
        print(args.dataset)
        raise ValueError("The parameter `dataset` should be one of [`mnist`, `fashionmnist`, `cifar10`, `cifar100`].")
    
    # Common initialization
    def collate_batch(batch): return torch.stack([x for x, _ in batch]).to(device), torch.tensor([y for _, y in batch]).to(device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    criterion = torch.nn.CrossEntropyLoss()

    # Train & test the CNN
    model = Net(in_features, out_features, n_cls).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
        test(test_loader, model, criterion)
        scheduler.step()

    # Train & test the LagrangeEmbedding-based network
    lagrange_model = LagrangeNet(train_loader, model.projection, dof=args.dof, n_cls=n_cls).to(device)
    lagrange_optimizer = torch.optim.Adadelta(lagrange_model.parameters(), lr=args.lr)
    lagrange_scheduler = torch.optim.lr_scheduler.StepLR(lagrange_optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(args.epochs):
        train(train_loader, lagrange_model, criterion, lagrange_optimizer, epoch, args.print_freq)
        test(test_loader, lagrange_model, criterion)
        lagrange_scheduler.step()
    print("Finish image classification!")


if __name__ == "__main__":
    main()
