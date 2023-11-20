from typing import Tuple, List, Union, Optional, Callable
import time
import torch


class LagrangeEmbedding(torch.nn.Module):
    @classmethod
    def init_mesh(cls, min_x: torch.Tensor, max_x: torch.Tensor, eps=0.1):
        """Generate a basic grid that covers all input points `x`."""
        assert torch.all(min_x < max_x)
        dim, = min_x.shape

        min_x, max_x = (1 + eps) * min_x - eps * max_x, (2 + 3 * eps) * max_x - (1 + 3 * eps) * min_x
        points = torch.tensor([[max_x[j].item() if i == j else min_x[j].item() for j in range(dim)] for i in range(dim + 1)], dtype=torch.float32, device=min_x.device)
        simplices = torch.arange(dim + 1, dtype=torch.int64, device=min_x.device).reshape(1, dim + 1)
        return points, simplices

    def compute_b2t(self, inv: torch.Tensor, x: torch.Tensor, count_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Convert point coordinates (x, y) to barycentric coordinates (u, v, w).
        # A point is considered inside the simplex if and only if all the barycentric coordinates are non-negative. 
        # [1] The tensor `x` containing point coordinates. It has shape [b, dim], where b is the number of points and dim is the number of dimensions.
        # [2] The tensor `shape_values` containing barycentric coordinates of points. It has shape [b, t, dim+1], where t is the number of simplices.
        shape_values = torch.einsum("tdv,bd->btv", inv[:, :-1, :], x) + inv[None, :, -1, :] # [b, t, dim+1]

        # The tensor `b2t` indicates the relationship between input points and simplices.  It has shape [b, t].
        # If b2t[i, j] is True, it signifies that the i-th point is located inside the j-th simplex.
        b2t = torch.all(shape_values >= 0, axis=-1)  # [b, t]
        
        # Do the multiply when using "LagrangeEmbedding.build_mesh_from_tensor".
        if count_x is not None:
            b2t = b2t * count_x[:, None]
        
        return torch.sum(b2t, dim=0, keepdim=True)  # [1, t]

    def sort_edges(self, b2t: torch.Tensor) -> torch.Tensor:
        """Consider an edge $E$ and let $S_E = {T_i | E \in T_i, i=1, 2, \cdots, t }$ be the set of simplices that contain edge $E$. 
        Furthermore, define $C(p, T) = # { p \in T | p \in P }$ and $\Xsi_{E, T} = 1 if E is the longest edge of T else 0$. 
        The evaluation of edge priority for splitting is as follows:
            \sum_{T \in S_E} (\Xsi_{E, T} * C(p, T)) / \sum_{T \in S_E} C(p, T)
        """
        p, dim = self._points.shape

        es2t = torch.stack([torch.min(self._simplices[:, [i, j]], dim=1)[0] for i in range(dim) for j in range(i + 1, dim + 1)], dim=0)  # [(dim+1)*dim/2, t]
        ee2t = torch.stack([torch.max(self._simplices[:, [i, j]], dim=1)[0] for i in range(dim) for j in range(i + 1, dim + 1)], dim=0)  # [(dim+1)*dim/2, t]
        elen = torch.linalg.norm(self._points[es2t, :] - self._points[ee2t, :], dim=2)  # [(dim+1)*dim/2, t]

        # Ignore the raw data count. If a simplex contains less than dim+1 raw data, the simplex satisfies the condition of the error-bound formula. 
        _b2t = torch.greater_equal(b2t, dim+1) * b2t

        indices = torch.stack([es2t.view(-1), ee2t.view(-1)], dim=0)
        numerator_values = ((torch.max(elen, dim=0)[0] == elen) * _b2t).view(-1)
        denominator_values = _b2t.repeat(ee2t.shape[0], 1).view(-1)
        numerator = torch.sparse_coo_tensor(indices, numerator_values, (p, p), device=self._points.device).coalesce()
        denominator = torch.sparse_coo_tensor(indices, denominator_values, (p, p), device=self._points.device).coalesce()

        ratios = numerator.values() / torch.maximum(denominator.values(), torch.ones_like(denominator.values()))
        # Ignore the edge that the amount of raw data located in the neighbour simplices is less than dim+1.
        max_ratio = torch.max(ratios)
        # If there are multiple ratios that reach the max values, select the max numerator.
        index = torch.argmax(torch.eq(ratios, max_ratio) * numerator.values())
        return numerator.indices()[:, index]

    def add_simplices(self, selected_edge: torch.Tensor) -> torch.Tensor:
        """Split a chosen edge and creating new simplices to refine the mesh."""
        p, _ = self._points.shape

        # Update `self._points`
        new_point = torch.mean(self._points[selected_edge, :], dim=0, keepdims=True)
        self._points = torch.cat([self._points, new_point], dim=0)

        # Update `self._simplices`
        drop_ids = [i for i, simplex in enumerate(self._simplices) if selected_edge[0] in simplex and selected_edge[1] in simplex]
        protected_simplices = [list(simplex) for i, simplex in enumerate(self._simplices) if i not in drop_ids]
        upper_simplices = [[p if point_id == selected_edge[1] else point_id for point_id in simplex] for i, simplex in enumerate(self._simplices) if i in drop_ids]
        lower_simplices = [[p if point_id == selected_edge[0] else point_id for point_id in simplex] for i, simplex in enumerate(self._simplices) if i in drop_ids]
        self._simplices = torch.tensor(protected_simplices + upper_simplices + lower_simplices).to(self._points.device)
        return selected_edge
    
    def build_mesh_from_dataloader(self, data_loader: torch.utils.data.dataloader.DataLoader, dof: int, pre_proc: Optional[Callable] = None, min_x: Optional[torch.Tensor] = None, max_x: Optional[torch.Tensor] = None):
        """Please set the first element in each batch as the `raw_data`. e.g., `images` is the `raw_data`, then `for images, labels in data_loader: ...`."""
        start_time = time.time()
        print("Initializing the input domain")
        pre_proc = pre_proc if pre_proc is not None else lambda x: x
        if min_x is None or max_x is None:
            min_x, max_x = None, None
            for batch in data_loader:
                batch_x = pre_proc(batch[0])
                if min_x is None or max_x is None:
                    min_x = torch.min(batch_x, axis=0)[0]
                    max_x = torch.max(batch_x, axis=0)[0]
                else:
                    min_x = torch.minimum(torch.min(batch_x, axis=0)[0], min_x)
                    max_x = torch.maximum(torch.max(batch_x, axis=0)[0], max_x)
        self._points, self._simplices = self.init_mesh(min_x=min_x, max_x=max_x)

        print("Refining the triangulation mesh")
        for _ in range(dof - self._simplices.shape[1]):
            # Compute inv
            coefficient_matrix = torch.nn.functional.pad(self._points[self._simplices, :], (0, 1), "constant", 1)  # [t, dim + 1, dim + 1]
            inv = torch.linalg.inv(coefficient_matrix)  # [t, dim + 1, dim + 1]
            # Compute b2t
            b2t = torch.zeros(size=(1, self._simplices.shape[0]), dtype=torch.int64, device=self._points.device)
            for batch in data_loader:
                batch_x = pre_proc(batch[0])
                b2t += self.compute_b2t(inv=inv, x=batch_x)  # [1, dim]
            assert torch.max(b2t) > self._simplices.shape[1]  # If false, the model will be overfit after training.
            assert dof >= self._simplices.shape[1]  # In the case of first-order element, the degree of freedom (dof) must be the number of nodes.
            # Select the priority edge
            selected_edge = self.sort_edges(b2t=b2t)
            # Refine the mesh
            selected_edge = self.add_simplices(selected_edge=selected_edge)
            # Display logs
            print("Adding a node to the middle of edge {} \t Total: {:4d} nodes {:4d} simplices".format(
                selected_edge.tolist(), self._points.shape[0], self._simplices.shape[0]), end="\r")
        print("\nTime of Mesh Refinement: {:.2f}s".format(time.time() - start_time))
    
    def build_mesh_from_tensor(self, data_distribution: Tuple[torch.Tensor, torch.Tensor], dof: int, pre_proc: Optional[Callable] = None, min_x: Optional[torch.Tensor] = None, max_x: Optional[torch.Tensor] = None):
        """`tensors` is a tuple of unique elements and their count. e.g., the `raw_data` = torch.tensor([A, B, C, A, B, A]), then `tensors = (torch.tensor([A, B, C]), torch.tensor([3, 2, 1]))`"""
        start_time = time.time()
        print("Initializing the input domain")
        pre_proc = pre_proc if pre_proc is not None else lambda x: x
        raw_x, count_x = data_distribution
        x = pre_proc(raw_x)
        min_x = torch.min(x, axis=0)[0]
        max_x = torch.max(x, axis=0)[0]
        self._points, self._simplices = self.init_mesh(min_x=min_x, max_x=max_x)

        print("Refining the triangulation mesh")
        for _ in range(dof - self._simplices.shape[1]):
            # Compute inv
            coefficient_matrix = torch.nn.functional.pad(self._points[self._simplices, :], (0, 1), "constant", 1)  # [t, dim + 1, dim + 1]
            inv = torch.linalg.inv(coefficient_matrix)  # [t, dim + 1, dim + 1]
            # Compute b2t
            b2t = self.compute_b2t(inv=inv, x=x, count_x=count_x)  # [1, dim]
            assert torch.max(b2t) > self._simplices.shape[1]  # If false, the model will be overfit after training.
            assert dof >= self._simplices.shape[1]  # In the case of first-order element, the degree of freedom (dof) must be the number of nodes.
            # Select the priority edge
            selected_edge = self.sort_edges(b2t=b2t)
            # Refine the mesh
            selected_edge = self.add_simplices(selected_edge=selected_edge)
            # Display logs
            print("Adding a node to the middle of edge {} \t Total: {:4d} nodes {:4d} simplices".format(
                selected_edge.tolist(), self._points.shape[0], self._simplices.shape[0]), end="\r")
        print("\nTime of Mesh Refinement: {:.2f}s".format(time.time() - start_time))

    def __init__(self, raw_data: Union[torch.utils.data.dataloader.DataLoader, Tuple[torch.Tensor, torch.Tensor]], dof:int, pre_proc: Optional[Callable] = None):
        """Compute first-order elements, then create basis functions."""
        super(LagrangeEmbedding, self).__init__()

        # Initial the mesh
        self._points, self._simplices = None, None
        if isinstance(raw_data, torch.utils.data.dataloader.DataLoader):
            self.build_mesh_from_dataloader(raw_data, dof, pre_proc=pre_proc)
        else:
            self.build_mesh_from_tensor(raw_data, dof, pre_proc=pre_proc)

        # Create basis functions
        # The tensor `coefficient_matrices` indicates the grid structure. It has shape [t, dim+1, dim+1].
        # coefficient_matrices[i, j, :-1] is the coordinates of the j-th vertex in the i-th simplex.
        assert dof == self._points.shape[0]  # In the case of first-order element, the degree of freedom (dof) must be the number of nodes
        t, _ = self._simplices.shape
        coefficient_matrices = torch.nn.functional.pad(self._points[self._simplices, :], (0, 1), "constant", 1)  # [t, dim + 1, dim + 1]
        self.inv = torch.linalg.inv(coefficient_matrices)  # [t, dim + 1, dim + 1]
        
        # The sparse tensor `p2t_mask` indicates the relationship between vertices and simplices. It has shape [p, t, dim+1].
        # If p2t_mask[i, j, k] is True, it signifies that the i-th node matches the k-th vertex of the j-th simplex.
        indices = torch.transpose(torch.tensor([[ip, it, list(ips).index(ip)] for ip in range(dof) for it, ips in enumerate(self._simplices) if ip in ips]), 0, 1)
        values = torch.ones(size=(indices.shape[1], ), dtype=torch.float32)
        size = [dof, t, self._simplices.shape[1]]
        self.p2t_mask = torch.sparse_coo_tensor(indices, values, size).to_dense()  # [p, t, dim+1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, dim] -> [N, p]
        if self.inv.device != x.device:
            self.inv = self.inv.to(x.device)
        if self.p2t_mask.device != x.device:
            self.p2t_mask = self.p2t_mask.to(x.device)

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
        return basis_values.detach()

    @classmethod
    def unit_test(cls, dof=512):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        np.random.seed(9)

        base_size = 64  # Number of values to generate
        radius = []
        for k in range(1, 4):
            radius.append(k * 2 * torch.pi + torch.randn(k ** 2 * base_size))
        radius = torch.cat(radius, dim=0)
        angles = 2 * torch.pi * torch.rand(radius.__len__())
        points = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

        # ---------- The mesh initialization by using PyTorch DataLoader
        dataset = torch.utils.data.TensorDataset(torch.tensor(points), )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        embed = LagrangeEmbedding(data_loader, dof=3)

        # Create a figure and an axis
        fig, ax = plt.subplots()
        # Initialize empty data for the animation
        data = []
        # Define a function to update the plot in each animation frame
        def update(frame):
            try:
                # Compute inv
                coefficient_matrix = torch.nn.functional.pad(embed._points[embed._simplices, :], (0, 1), "constant", 1)  # [t, dim + 1, dim + 1]
                inv = torch.linalg.inv(coefficient_matrix)  # [t, dim + 1, dim + 1]

                # Terminate the mesh refinement when all simplices contain at least dim+1 raw data.
                b2t = torch.zeros(size=(1, embed._simplices.shape[0]), dtype=torch.int64)
                bz = 32
                for i in range(points.shape[0] // bz):
                    b2t += embed.compute_b2t(inv=inv, x=points[i*bz:(i+1)*bz, :])  # [1, dim]
                assert torch.max(b2t) > embed._simplices.shape[1]

                # Select the priority edge
                selected_edge = embed.sort_edges(b2t=b2t)

                # Refine the mesh
                selected_edge = embed.add_simplices(selected_edge=selected_edge)
                
                # Display logs
                print("Adding a node to the middle of edge {} \t Total: {:4d} nodes {:4d} simplices".format(
                    selected_edge.tolist(), embed._points.shape[0], embed._simplices.shape[0]), end="\r")
                
                ax.clear()  # Clear the axis
                data.append(frame)  # Append data for this frame
                ax.scatter(points[:, 0], points[:, 1])
                ax.triplot(embed._points[:, 0], embed._points[:, 1], triangles=embed._simplices, c='red', linewidth=0.5)
            except AssertionError:
                pass
        ani = FuncAnimation(fig, update, frames=range(dof), repeat=False, interval=50)
        # Save the animation as a GIF file
        ani.save('animation.gif', writer='pillow')
        # Display the animation (optional)
        plt.show()


class Net(torch.nn.Module):
    def pre_proc(self, raw_data, **kwargs) -> torch.Tensor:
        return raw_data
    
    def post_proc(self, x: torch.Tensor, *args) -> torch.Tensor:
        return x
    
    def __init__(self, data_loader: torch.utils.data.dataloader.DataLoader, dof: int, output_size=1):
        super(Net, self).__init__()

        self.backbone = LagrangeEmbedding(data_loader, dof, pre_proc=self.pre_proc)
        self.fc = torch.nn.Linear(dof, output_size, bias=False)

    def forward(self, x):
        x = self.pre_proc(x)  # [N, ?] -> [N, ?]
        basis_values = self.backbone(x)  # [N, ?] -> [N, dof]
        x = self.post_proc(basis_values)  # [N, dof] -> [N, dof]
        return self.fc(x)
    
    @classmethod
    def unit_test_1(cls, dof=128, num_epochs=5, num_samples=3000, device="cuda"):
        import numpy as np

        _min_x, _max_x = 0.02, 0.5
        nodes = 1 / ((1 / _min_x - 1 / _max_x) * np.random.rand(2*num_samples, 1) + 1 / _max_x)
        values = np.random.normal(loc=np.sin(1 / nodes), scale=0.5 * (nodes ** 2), size=(2*num_samples, 1))
        (x_train, y_train) = np.asarray(nodes[:num_samples], dtype=np.float32), np.asarray(values[:num_samples], dtype=np.float32)
        (x_test, y_test) = np.asarray(nodes[num_samples:], dtype=np.float32), np.asarray(values[num_samples:], dtype=np.float32)

        print("Loading data")
        x_train_tensor = torch.tensor(x_train, device=device)
        y_train_tensor = torch.tensor(y_train, device=device)
        x_test_tensor = torch.tensor(x_test, device=device)
        y_test_tensor = torch.tensor(y_test, device=device)
        batch_size = 200
        data = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

        print("Creating model")
        model = Net(data_loader, dof, output_size=1).to(device)

        # Count the number of trainable parameters
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters: {num_trainable_params}")

        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        print("Start training")
        # Training loop
        for epoch in range(num_epochs):
            # Set the model to training mode
            model.train()

            # Mini-batch training
            for x_batch, y_batch in data_loader:
                # Extract mini-batches
                inputs = x_batch.to(device)
                targets = y_batch.to(device)

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Set the model to evaluation mode
            model.eval()
            
            # Evaluation on the test set
            with torch.no_grad():
                mse = 0
                for i in range(0, x_test_tensor.size(0), batch_size):
                    # Extract mini-batches
                    inputs = x_test_tensor[i:i+batch_size].to(device)
                    targets = y_test_tensor[i:i+batch_size].to(device)
                    outputs = model(inputs)
                    mse += torch.sum(torch.square(outputs - targets))
                mse /= x_test_tensor.size(0) // batch_size
                clr = optimizer.param_groups[0]['lr']

            # Print training progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test MSE: {mse:.4f}, Current LR: {clr:.4f}")

        from matplotlib import pyplot as plt
        plt.scatter(x_train.flatten(), y_train.flatten())
        x_flatten = np.linspace(0.02, 1.0, num=1000)
        y_flatten = model(torch.tensor(x_flatten[:, None], dtype=torch.float32).to(device)).detach().cpu().numpy().flatten()
        plt.plot(x_flatten, y_flatten, c='r')
        plt.show()
    
    @classmethod
    def unit_test_2(cls, dof=512, num_epochs=5, num_samples=10000, device="cuda"):
        import numpy as np

        input_examples = 2 * np.pi * np.random.rand(2*num_samples, 2)
        output_examples = np.sin(input_examples[:, [0]]) + np.cos(input_examples[:, [1]])

        (x_train, y_train) = np.asarray(input_examples[:num_samples], dtype=np.float32), np.asarray(output_examples[:num_samples], dtype=np.float32)
        (x_test, y_test) = np.asarray(input_examples[num_samples:], dtype=np.float32), np.asarray(output_examples[num_samples:], dtype=np.float32)

        print("Loading data")
        x_train_tensor = torch.tensor(x_train, device=device)
        y_train_tensor = torch.tensor(y_train, device=device)
        x_test_tensor = torch.tensor(x_test, device=device)
        y_test_tensor = torch.tensor(y_test, device=device)
        batch_size = 100
        data = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

        print("Creating model")
        model = Net(data_loader, dof, output_size=1).to(device)

        # Count the number of trainable parameters
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters: {num_trainable_params}")

        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        print("Start training")
        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode

            # Mini-batch training
            for x_batch, y_batch in data_loader:
                # Extract mini-batches
                inputs = x_batch.to(device)
                targets = y_batch.to(device)

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            
            # Set the model to evaluation mode
            model.eval()
            
            # Evaluation on the test set
            with torch.no_grad():
                mse = 0
                for i in range(0, x_test_tensor.size(0), batch_size):
                    # Extract mini-batches
                    inputs = x_test_tensor[i:i+batch_size].to(device)
                    targets = y_test_tensor[i:i+batch_size].to(device)
                    outputs = model(inputs)
                    mse += torch.sum(torch.square(outputs - targets))
                mse /= x_test_tensor.size(0) // batch_size
                clr = optimizer.param_groups[0]['lr']

            # Print training progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test MSE: {mse:.4f}, Current LR: {clr:.4f}")

        from matplotlib import pyplot as plt
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.linspace(0, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        x_tensor = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
        y_tensor = []
        for i in range(0, x_test_tensor.size(0), batch_size):
            inputs = x_tensor[i:i+batch_size].to(device)
            outputs = model(inputs)
            y_tensor.append(outputs)
        y_tensor = torch.concat(y_tensor, dim=0)
        Z = np.reshape(y_tensor.detach().cpu().numpy(), X.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.show()


if __name__ == "__main__":
    LagrangeEmbedding.unit_test()
    # Net.unit_test_1(device="cpu")
    # Net.unit_test_1(device="cuda")
    # Net.unit_test_2(device="cpu")
    # Net.unit_test_2(device="cuda")
