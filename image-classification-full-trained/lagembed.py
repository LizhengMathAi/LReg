from typing import Callable, Tuple, Optional
import types
import math
import torch
import torch.nn as nn


class SigmoidNormalization(torch.nn.Module):
    def __init__(self, in_channels: int, n_components: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SigmoidNormalization, self).__init__()

        self.weight = nn.Parameter(torch.empty((n_components, in_channels), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(n_components, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.einsum("...i,ni->...n", x, self.weight) + self.bias  # [B, ..., in_channels], [n_components, in_channels], [n_components] -> [B, ..., n_components]
        return torch.nn.functional.sigmoid(x)


class LagEncoder(torch.nn.Module):
    def __init__(self, n_components: int, dof: int):
        super(LagEncoder, self).__init__()

        self.register_buffer("bin_edges", torch.linspace(-1., 1., dof)[None, :].repeat(n_components, 1))  # [n_components, dof]
        self.register_buffer('counts', torch.zeros((n_components, dof - 1), dtype=torch.long))  # [n_components, dof-1]

    def forward(self, x):
        with torch.no_grad():
            # The LagrangeEncoder by using P1(R1) basis functions.  TODO: Implementation of general basis functions is needed.
            offset = x[..., None] - self.bin_edges  # [B, ..., n_components], [n_components, dof] -> [B, ..., n_components, dof] TODO: unsafe!
            scale = 1 / (self.bin_edges[:, 1:] - self.bin_edges[:, :-1])  # [n_components, dof-1]
            left_x = scale * nn.functional.relu(offset[..., :-1])  # [B, ..., n_components, dof-1] TODO: unsafe!
            right_x = scale * nn.functional.relu(-offset[..., 1:])  # [B, ..., n_components, dof-1] TODO: unsafe!
            x = torch.concat([right_x[..., [0]], torch.minimum(left_x[..., :-1], right_x[..., 1:]), left_x[..., [-1]]], dim=-1)  # [B, ..., n_components, dof]

        return x

    @classmethod
    def count_intervals(cls, coarse_nodes: torch.Tensor, inputs: torch.Tensor, losses: torch.Tensor, topk=1) -> torch.Tensor:
        """
        Select inputs of the top-k loss value, then count how many input values fall into each interval defined by 
        consecutive elements in each row of coarse_nodes.

        Args:
            coarse_nodes (torch.Tensor): A 2D tensor where each row represents sorted thresholds defining intervals.
            inputs (torch.Tensor): A 2D tensor where each row represents input values to be counted within the intervals.
            losses (torch.Tensor): A 1D tensor where each value represents a loss of input.
            topk (int): A integer for selecting top-k losses

        Returns:
            indices (torch.Tensor): A 2D tensor with shape [dim, dof-1], the counts of the intervals should be refined.

        Example:
            >>> coarse_nodes = torch.tensor([
                    [1, 2, 4, 6, 8],
                    [1.5, 3.5, 5.5, 7.5, 9.5]
                ], dtype=torch.float32)
            >>> inputs = torch.tensor([
                    [2, 4],
                    [5, 2],
                    [5, 4],
                    [3, 7],
                    [1, 8],
                ], dtype=torch.float32)
            >>> losses = torch.tensor([1.0, 0.2, 0.1, 1.5, 0.5], dtype=torch.float32)
            >>> topk = 3
            >>> counts = RemainderModel.count_intervals(coarse_nodes, inputs, losses, topk)
            >>> counts
            tensor([[1, 2, 0, 0],
                    [0, 1, 1, 1]])
        """
        assert inputs.shape[0] == losses.shape[0]
        topk_losses, topk_indices = torch.topk(losses, k=topk, dim=-1, largest=True, sorted=False)  # [topk, ]

        inputs = torch.flatten(inputs, start_dim=1)  # [B, in_channels]
        cum_counts = torch.sum(torch.greater_equal(inputs[topk_indices, :, None], coarse_nodes[None, :, :]).long(), dim=0)  # [in_channels, dof]
        counts = cum_counts[:, :-1] - cum_counts[:, 1:]  # [in_channels, dof-1]

        return counts

    @classmethod
    def select_intervals(cls, counts: torch.Tensor, topk: int) -> torch.Tensor:
        """
        Select intervals of the top-k count and return the indices of these intervals.

        Args:
            counts (torch.Tensor): A 2D tensor with shape [dim, dof-1] represents the counts of the intervals that should be refined.
            topk (int): A integer for selecting top-k counts

        Returns:
            indices (torch.Tensor): A 2D tensor with shape [dim, topk], the indices of the intervals which should be refined.

        Example:
            >>> counts = torch.tensor([
                    [1, 2, 0, 0],
                    [0, 1, 1, 1],
                ], dtype=torch.long)
            >>> topk = 2
            >>> indices = RemainderModel.select_intervals(counts, topk)  # [in_channels, topk]
            >>> indices
            tensor([[1, 0],
                    [2, 3]])
        """
        values, indices = torch.topk(counts, k=topk, dim=-1, largest=True, sorted=False)  # [in_channels, topk], [in_channels, topk]
        return torch.sort(indices, dim=-1).values

    @classmethod
    def refinement(cls, coarse_counts: torch.Tensor, coarse_nodes: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refines a mesh by inserting midpoints based on specified indices.

        Parameters:
            coarse_counts (torch.Tensor): A 2D tensor where each row represents counts corresponding to intervals.
            coarse_nodes (torch.Tensor): A 2D tensor where each row represents a set of nodes.
            indices (torch.Tensor): A 2D tensor of indices indicating positions between which midpoints should be inserted.

        Returns:
            fine_nodes: The nodes after insertion of midpoints.
            fine_counts: The counts after insertion of midpoints.

        Example:
            >>> coarse_counts = torch.tensor([
                    [1, 2, 0, 0],
                    [0, 1, 1, 1]
                ], dtype=torch.float32)
            >>> coarse_nodes = torch.tensor([
                    [1, 2, 4, 6, 8],
                    [1.5, 3.5, 5.5, 7.5, 9.5]
                ], dtype=torch.float32)
            >>> indices = torch.tensor([
                    [0, 2, 3],
                    [0, 1, 3]
                ], dtype=torch.int64)
            >>> fine_counts, fine_nodes = RemainderModel.refinement(coarse_counts, coarse_nodes, indices)
            >>> fine_counts
            tensor([[0.5000, 0.5000, 2.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.5000, 0.5000, 1.0000, 0.5000, 0.5000]])
            >>> fine_nodes
            tensor([[1.0000, 1.5000, 2.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000],
                    [1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 7.5000, 8.5000, 9.5000]])
        """
        fine_counts = []
        fine_nodes = []

        # Iterate through each row of nodes, coefficients, and indices
        for row_nodes, row_counts, row_ids in zip(coarse_nodes, coarse_counts, indices):
            current_counts = row_counts.clone()
            current_nodes = row_nodes.clone()

            # Convert row_ids to integer type suitable for indexing
            row_ids = row_ids.long()

            offset = 0
            for idx in row_ids:
                idx = idx.item()  # Convert to Python int for indexing adjustments
                mid_count = current_counts[idx + offset] / 2
                mid_node = (current_nodes[idx + offset] + current_nodes[idx + offset + 1]) / 2

                # Slicing and concatenating to simulate insert
                current_counts = torch.cat((current_counts[:idx + offset], mid_count.unsqueeze(0), mid_count.unsqueeze(0), current_counts[idx + offset + 1:]))
                current_nodes = torch.cat((current_nodes[:idx + offset + 1], mid_node.unsqueeze(0), current_nodes[idx + offset + 1:]))

                # Update offset because the tensor size has increased
                offset += 1

            fine_counts.append(current_counts)
            fine_nodes.append(current_nodes)

        # Stack all rows to create a single tensor for nodes and coefficients
        fine_counts = torch.stack(fine_counts)
        fine_nodes = torch.stack(fine_nodes)

        return fine_counts, fine_nodes

    @classmethod
    def to_equal_freq(cls, in_values: torch.Tensor, in_bin_edges: torch.Tensor, dof: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform a given distribution into an equal-frequency distribution.

        This method normalizes input histogram values to a Probability Density Function (PDF), computes the Cumulative
        Distribution Function (CDF), and then redefines bin edges such that each bin in the resultant histogram has an
        equal total probability, thereby creating an equal-frequency distribution.

        Args:
            in_values (torch.Tensor): A 2D tensor of shape [dim, dof'-1], representing the values in each bin of the 
                                      input distribution.
            in_bin_edges (torch.Tensor): A 2D tensor of shape [dim, dof'], representing the edges of the bins in the 
                                         input distribution.
            dof (Optional[int]): The desired degrees of freedom for the output distribution (one more than the number
                                 of bins). If None, the dof of the input distribution is used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - out_values (torch.Tensor): A 2D tensor with shape [dim, dof-1], representing the values of each bin
                                             in the equal-frequency distribution.
                - out_bin_edges (torch.Tensor): A 2D tensor with shape [dim, dof], representing the edges of the bins
                                                in the equal-frequency distribution.


        Example:
            >>> out_values = torch.tensor([
                [1.0000, 4.0000, 3.0000, 2.0000],
                [1.0000, 4.0000, 3.0000, 2.0000]
            ], dtype=torch.float32)
            >>> out_bin_edges = torch.tensor([
                [0.5000, 0.6000, 0.7000, 0.8000, 0.9000],
                [0.6000, 0.7000, 0.8000, 0.9000, 1.0000]
            ], dtype=torch.float32)
            >>> in_values, in_bin_edges = RemainderModel.to_equal_freq(out_values, out_bin_edges)

            >>> in_values
            tensor([[1.8182, 4.0000, 3.0000, 2.1429],
                    [1.8182, 4.0000, 3.0000, 2.1429]])
            >>> in_bin_edges
            tensor([[0.5000, 0.6375, 0.7000, 0.7833, 0.9000],
                    [0.6000, 0.7375, 0.8000, 0.8833, 1.0000]])

        The method first computes the normalized PDF from the input values and bin edges, then calculates the CDF. It 
        uses the CDF to define new bin edges, ensuring equal cumulative probability in each bin of the new distribution. 
        The output values are then derived based on these new bin edges, maintaining an equal frequency across bins.
        """
        if dof is None:
            dof = in_bin_edges.shape[1]

        scale = torch.sum((in_bin_edges[:, 1:] - in_bin_edges[:, :-1]) * in_values, dim=1, keepdim=True)  # [in_channels, 1]
        in_pdf = in_values / scale  # [in_channels, dof-1]
        in_cdf = torch.cumsum((in_bin_edges[:, 1:] - in_bin_edges[:, :-1]) * in_pdf, dim=1)  # [in_channels, dof-1]
        in_cdf = torch.concatenate([torch.zeros_like(in_cdf[:, [0]]), in_cdf], dim=1)  # [in_channels, dof]

        out_cdf = torch.linspace(0., 1., dof, device=in_bin_edges.device) # [dof', ]
        out_bin_edges = (in_bin_edges[:, 1:, None] * (out_cdf[None, None, 1:-1] - in_cdf[:, :-1, None]) - 
                        in_bin_edges[:, :-1, None] * (out_cdf[None, None, 1:-1] - in_cdf[:, 1:, None])
                        ) / (in_cdf[:, 1:, None] - in_cdf[:, :-1, None])  # [in_channels, dof-1, dof'-2]
        ids = torch.argmax((out_bin_edges >= in_bin_edges[:, :-1, None]).to(torch.float32) * 
                        (out_bin_edges < in_bin_edges[:, 1:, None]).to(torch.float32), dim=1, keepdim=True) # [in_channels, 1, dof'-2]
        out_bin_edges = torch.squeeze(torch.take_along_dim(out_bin_edges, ids, dim=1), dim=1)  # [in_channels, dof'-2]
        out_bin_edges = torch.concatenate([in_bin_edges[:, [0]], out_bin_edges], dim=1)  # [in_channels, dof'-1]
        out_bin_edges = torch.concatenate([out_bin_edges, in_bin_edges[:, [-1]]], dim=1)  # [in_channels, dof']

        out_values = scale / (dof - 1) / (out_bin_edges[:, 1:] - out_bin_edges[:, :-1])
        
        return out_values.detach(), out_bin_edges.detach()

    @classmethod
    def coarsening(cls, fine_counts: torch.Tensor, fine_nodes: torch.Tensor, dof: int, eps=1) -> torch.Tensor:
        """
        Transform a set of scatters `fine_nodes` into a new set of scatters `coarse_nodes` which has similar 
        distribution.
        
        Args:
            fine_counts (torch.Tensor): A 2D tensor of shape [dim, dof'-1], representing the weights of intervals
                                        in the fine mesh.
            fine_nodes (torch.Tensor): A 2D tensor of shape [dim, dof'], representing the edges of intervals of
                                       the fine mesh.
            dof (Optional[int]): The desired degrees of freedom for the coarse mesh (one more than the number of 
                                 intervals). If None, the dof of the fine mesh will be used.
            eps (int): 

        Returns:
            out_bin_edges (torch.Tensor): A 2D tensor with shape [dim, dof], representing the edges of intervals
                                          of the coarse mesh.

        Example:
            >>> fine_nodes = torch.tensor([
                [1.0000, 1.5000, 2.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000],
                [1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 7.5000, 8.5000, 9.5000]
            ], dtype=torch.float32)
            >>> fine_counts = torch.tensor([
                [0.5000, 0.5000, 2.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000, 0.5000, 1.0000, 0.5000, 0.5000]
            ], dtype=torch.float32)
            >>> coarse_nodes = RemainderModel.coarsening(fine_counts, fine_nodes, dof=4, eps=0)
            >>> coarse_nodes
            tensor([[1.0000, 2.0000, 3.0000, 8.0000],
                    [1.5000, 5.5000, 7.5000, 9.5000]])
        """
        in_values = (fine_counts + eps) / (fine_nodes[:, 1:] - fine_nodes[:, :-1])
        in_bin_edges = fine_nodes

        out_values, out_bin_edges = cls.to_equal_freq(in_values, in_bin_edges, dof)  # [in_channels, dof-1], [in_channels, dof]
        return out_bin_edges

    @classmethod
    def linear_interpolation(cls, x: torch.Tensor, x_values: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:
        """
        Perform linear interpolation at points x, given the data points (x_values, y_values).
        
        Parameters:
            x: Points at which to interpolate.
            x_values: X-coordinates of the data points.
            y_values: Y-coordinates of the data points.
        
        Returns:
            float or tensor-like: Interpolated values corresponding to points x.

        Example data
            import matplotlib.pyplot as plt
            
            x_values = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.float32)  # [in_channels dof]
            y_values = torch.tensor([[1, 2, 4, 3, 5], [0, 1, 3, 2, 4]], dtype=torch.float32)  # [in_channels, dof]

            # Interpolation points
            x_to_interpolate = torch.stack([torch.linspace(0, 4, 100), torch.linspace(1, 5, 100)], dim=1)  # [B, in_channels]
            interpolated_values = RemainderModel.linear_interpolation(x_to_interpolate, x_values, y_values[:, :, None])[:, :, 0]

            # Plot original data and interpolated curve
            plt.plot(x_values.numpy().flatten(), y_values.numpy().flatten(), 'o', label='Original Data')
            plt.plot(x_to_interpolate.numpy()[:, 0], interpolated_values.numpy()[:, 0], label='Interpolated Curve 1')
            plt.plot(x_to_interpolate.numpy()[:, 1], interpolated_values.numpy()[:, 1], label='Interpolated Curve 2')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('1D Linear Interpolation')
            plt.legend()
            plt.grid(True)
            plt.show()
        """
        # The LagrangeEncoder by using P1(R1) basis functions.  TODO: Implementation of general basis functions is needed.
        offset = x[:, :, None] - x_values[None, :, :]  # [B, in_channels], [in_channels, dof] -> [B, in_channels, dof]
        scale = 1 / (x_values[None, :, 1:] - x_values[None, :, :-1])  # [1, in_channels, dof-1]
        left_x = scale * nn.functional.relu(offset[:, :, :-1])  # [B, in_channels, dof-1]
        right_x = scale * nn.functional.relu(-offset[:, :, 1:])  # [B, in_channels, dof-1]
        x = torch.concat([right_x[:, :, [0]], torch.minimum(left_x[:, :, :-1], right_x[:, :, 1:]), left_x[:, :, [-1]]], dim=2)  # [B, dim, dof]

        return torch.einsum("bid,ido->bio", x, y_values)  # [B, in_channels, dof], [in_channels, dof, out_channels] -> [B, in_channels, out_channels]

    def update_mesh(self, coeff, topk):
        dof = self.counts.shape[1] + 1

        # generate fine nodes
        indices = self.select_intervals(self.counts, topk)  # [in_channels, dof-1] -> [in_channels, topk]
        fine_counts, fine_nodes = self.refinement(self.counts, self.bin_edges, indices)  # [in_channels, dof + topk - 1], [in_channels, dof + topk]
        
        # reset fine mesh to a new coarse mesh
        bin_edges = self.coarsening(fine_counts, fine_nodes, dof)  # [in_channels, dof + topk] -> [in_channels, dof]

        # linear interpolation, compute the new coeff on the new coarse mesh
        x_values = torch.transpose(bin_edges, dim0=1, dim1=0)  # [dof, in_channels]
        y_values = self.linear_interpolation(x_values, self.bin_edges, coeff)  # [dof, in_channels, out_channels]
        updated_coeff = torch.transpose(y_values, dim0=1, dim1=0)  # [in_channels, dof, out_channels]

        # updating parameters
        self.bin_edges = bin_edges.detach()
        self.counts.zero_()
        return updated_coeff


class LagHead(torch.nn.Module):
    def __init__(self, n_components: int, out_channels: int, dof: int):
        super(LagHead, self).__init__()

        self.coeff = nn.Parameter(torch.zeros((n_components, dof, out_channels)))  # [n_components, dof, out_channels]

    def forward(self, x):
        return torch.einsum("...id,ido->...o", x, self.coeff)  # [B, ..., n_components, dof], [n_components, dof, out_channels] -> [B, ..., out_channels]


class RemainderModel(nn.Module):
    def __init__(self, in_shape: Tuple[int], n_components:int, out_shape: Tuple[int], dof: int, channel_dim: int = 1):
        super(RemainderModel, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.channel_dim = channel_dim

        self.sn = SigmoidNormalization(in_channels=in_shape[self.channel_dim], n_components=n_components)
        self.encoder = LagEncoder(n_components=n_components, dof=dof)
        self.head = LagHead(n_components=n_components, out_channels=out_shape[self.channel_dim], dof=dof)

    def preprocess(self, x):
        if x.dim() > 2:
            dims = list(range(self.channel_dim)) + list(range(self.channel_dim+1, x.dim())) + [self.channel_dim]
            x = torch.permute(x, dims=dims)
        return x
    
    def postprocess(self, x):
        if x.dim() > 2:
            dims = list(range(self.channel_dim)) + [x.dim()-1] + list(range(self.channel_dim, x.dim()-1))
            x = torch.permute(x, dims=dims)

        if x.dim() == 1:
            pass
        elif x.dim() == 2:
            pass
        elif x.dim() == 3:
            _, _, in_L = self.in_shape
            _, _, out_L = self.out_shape
            
            # Compute the kernel size
            kernel_size = math.ceil(in_L / out_L)

            # Compute the stride
            stride = math.floor(in_L / out_L)
            
            # Calculate the required padding for height and width
            padding = max((out_L - 1) * stride + kernel_size - in_L, 0)
            padding = (padding // 2, padding - padding // 2)
            padding = padding if sum(padding) > 0 else 0
            
            # Use avg_pool with calculated kernel, stride, and padding
            x = torch.nn.functional.avg_pool1d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        elif x.dim() == 4:
            _, _, in_H, in_W = self.in_shape
            _, _, out_H, out_W = self.out_shape
            
            # Compute the kernel size
            kernel_size_h = math.ceil(in_H / out_H)
            kernel_size_w = math.ceil(in_W / out_W)
            kernel_size = (kernel_size_h, kernel_size_w)

            # Compute the stride
            stride_h = math.floor(in_H / out_H)
            stride_w = math.floor(in_W / out_W)
            stride=(stride_h, stride_w)
            
            # Calculate the required padding for height and width
            padding_h = max((out_H - 1) * stride_h + kernel_size_h - in_H, 0)
            padding_w = max((out_W - 1) * stride_w + kernel_size_w - in_W, 0)
            padding = (padding_w // 2, padding_w - padding_w // 2, padding_h // 2, padding_h - padding_h // 2)
            padding = padding if sum(padding) > 0 else 0
            
            # Use avg_pool with calculated kernel, stride, and padding
            x = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        elif x.dim() == 5:
            _, _, in_D, in_H, in_W = self.in_shape
            _, _, out_D, out_H, out_W = self.out_shape
            
            # Compute the kernel size
            kernel_size_d = math.ceil(in_D / out_D)
            kernel_size_h = math.ceil(in_H / out_H)
            kernel_size_w = math.ceil(in_W / out_W)
            kernel_size = (kernel_size_d, kernel_size_h, kernel_size_w)

            # Compute the stride
            stride_d = math.floor(in_D / out_D)
            stride_h = math.floor(in_H / out_H)
            stride_w = math.floor(in_W / out_W)
            stride=(stride_d, stride_h, stride_w)
            
            # Calculate the required padding for height and width
            padding_d = max((out_D - 1) * stride_d + kernel_size_d - in_D, 0)
            padding_h = max((out_H - 1) * stride_h + kernel_size_h - in_H, 0)
            padding_w = max((out_W - 1) * stride_w + kernel_size_w - in_W, 0)
            padding = (padding_d // 2, padding_d - padding_d // 2, padding_w // 2, padding_w - padding_w // 2, padding_h // 2, padding_h - padding_h // 2)
            padding = padding if sum(padding) > 0 else 0
            
            # Use avg_pool with calculated kernel, stride, and padding
            x = torch.nn.functional.avg_pool3d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError('Aligning shape failed.')
        
        return x


    def forward(self, x):
        x = self.preprocess(x)

        x = self.sn(x)
        x = self.encoder(x)
        x = self.head(x)

        x = self.postprocess(x)

        return x
    
    def compute_l1(self):
        avg = torch.mean((self.head.coeff[:, 2:] + self.head.coeff[:, :-2]) / 2, dim=0, keepdim=True)  # NOTE: window size = 3
        reg = torch.mean(torch.abs(self.head.coeff[:, 1:-1] - avg))  # NOTE: L1-regularization of the first-order gradients
        return reg
    
    def update_encoder(self, topk):
        self.head.coeff.data = self.encoder.update_mesh(self.head.coeff.data, topk=topk)


def add_remainder(model: nn.Module, forward: Callable, in_shape: Tuple[int], n_components: int, out_shape: Tuple[int], dof: int, channel_dim: int=1) -> nn.Module:
    """
    Add a remainder branch to a given model, integrate custom layers, and modify its forward method.

              +-----+                             +-----+                                  +-----+
    input --> | ... | --+-----------------------> | ... | ---------------------------+---> | ... | --> output
              +-----+   |                         +-----+                            |     +-----+
                        |                                                            |
                        |    +---------------------------------------------------+   |
                        +--> |                   remainder                       | --+
                             | +-----+     +------+     +---------+     +------+ |
                             | | PCA | --> | Norm | --> | Encoder | --> | Head | |
                             | +-----+     +------+     +---------+     +------+ |
                             +---------------------------------------------------+

    Args:
        model (nn.Module): The original model to which the remainder branch is added.
        forward (Callable): A function that defines the backbone part of the model.
        train_loader (DataLoader): DataLoader used for training the model.
        dof (int): Degree of freedom for initializing the remainder mesh.
        out_channels (int): Output channels of the remainder model.
        none_reduction_criterion: Loss function with parameter `reduction="none"`.
        topk (int): A integer for selecting top-k counts
        rr (float): A rate for regularization item

    Returns:
        nn.Module: The modified model with the remainder branch.
    """    
    device = next(model.parameters()).device
    remainder = RemainderModel(in_shape, n_components, out_shape, dof, channel_dim).to(device)
    print("Number of added parameters:", sum([p.numel() for p in remainder.parameters()]))

    model.forward = types.MethodType(forward, model)
    # model.remainder = remainder
    if hasattr(model, "remainder"):
        model.remainder.append(remainder)
    else:
        setattr(model, "remainder", torch.nn.ModuleList([remainder]))

    model.named_children

    # Overwrite the `train` method
    def train(self, mode=True):
        super(self.__class__, self).train(mode)  # Call the parent class train method
        # if mode:  # Freeze the original model and unfreeze the remainder branch
        #     for name, module in self.named_children():
        #         if name == "remainder":
        #             module.eval()
        #             model[-1].train()
        #         else:
        #             module.eval()
        # else:
        #     for name, module in self.named_children():
        #         module.eval()
        if mode:  # Freeze the original model and unfreeze the remainder branch
            for name, module in self.named_children():
                module.eval()
            self.remainder[-1].train()

    model.train = types.MethodType(train, model)

    return model


def mobilenet_v2_wrapper(model: nn.Module, n_components: int, dof: int):
    # Baseline: "acc@1": 71.878, "acc@5": 90.286
    in_shape, out_shape, channel_dim = (None, 64, 14, 14), (None, 1280, 7, 7), 1

    # Overwrite the `forward` method
    # Example:
    #     `n_components`: 16, `dof`: 16 `lr`: 0.01
    #     Number of parameters: 328720
    #     Acc@1 71.914 Acc@5 90.230
    def forward(self, x):
        for k, module in enumerate(self.features):
            if k == 11:
                re = sum([module(x) for module in self.remainder])
            x = module(x)
        x = x + re
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)        
        output = self.classifier(x)
        
        reg = self.remainder[-1].compute_l1()

        return output, reg

    return add_remainder(model, forward, in_shape, n_components, out_shape, dof, channel_dim)



def resnet_wrapper(model: nn.Module, n_components: int, dof: int):
    # Baseline: "acc@1": 76.130, "acc@5": 92.862
    in_shape, out_shape, channel_dim = (None, 1024, 14, 14), (None, 2048, 7, 7), 1

    # Overwrite the `forward` method
    # Example:
    #     `n_components`: 16, `dof`: 16 `lr`: 0.01
    #     Number of parameters: 540688
    #     Acc@1 76.202 Acc@5 92.972
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) + sum([module(x) for module in self.remainder])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        
        reg = self.remainder[-1].compute_l1()

        return output, reg
        
    return add_remainder(model, forward, in_shape, n_components, out_shape, dof, channel_dim)


def resnext_wrapper(model: nn.Module, n_components: int, dof: int):
    # Baseline: "acc@1": 77.618, "acc@5": 93.698
    in_shape, out_shape, channel_dim = (None, 1024, 14, 14), (None, 2048, 7, 7), 1

    # Overwrite the `forward` method
    # Example:
    #     `n_components`: 16, `dof`: 16 `lr`: 0.001
    #     Number of parameters: 540688
    #     Acc@1 77.626 Acc@5 93.676
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) + sum([module(x) for module in self.remainder])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        
        reg = self.remainder[-1].compute_l1()

        return output, reg
        
    return add_remainder(model, forward, in_shape, n_components, out_shape, dof, channel_dim)


def efficientnet_wrapper(model: nn.Module, in_channels: int, n_components: int, dof: int, criterion):

    def backbone(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def head(self, x):
        x = self.classifier(x)
        return x

    return add_remainder(model, backbone, head, in_channels, n_components, 1000, dof, criterion, topk=1, rr=1e-4)


def transformer_wrapper(model: nn.Module, n_components: int, dof: int):
    # Baseline: "acc@1": 81.072, "acc@5": 95.318
    in_shape, out_shape, channel_dim = (None, 768), (None, 1000), 1

    # Overwrite the `forward` method
    # Example:
    #     `n_components`: 16, `dof`: 16 `lr`: 0.0003
    #     Number of parameters: 208912
    #     Acc@1 81.076 Acc@5 95.322
    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        
        output = self.heads(x) + sum([module(x) for module in self.remainder])
        
        # reg = self.remainder[-1].compute_l1()

        return output, 0.
    
    return add_remainder(model, forward, in_shape, n_components, out_shape, dof, channel_dim)


def wrapper(name: str, model: nn.Module, n_components: int, dof: int):
    if name == "mobilenet_v2":
        return mobilenet_v2_wrapper(model, n_components, dof)
    elif name == "resnet18":
        return resnet_wrapper(model, 512, n_components, dof)
    elif name == "resnet50":
        return resnet_wrapper(model, n_components, dof)
    elif name == "resnext50_32x4d":
        return resnext_wrapper(model, n_components, dof)
    elif name in ["efficientnet_v2_s", "efficientnet_v2_m"]:
        return efficientnet_wrapper(model, 1280, n_components, dof)
    elif name in ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]:
        return transformer_wrapper(model, n_components, dof)
    else:
        raise ValueError

