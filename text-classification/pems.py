from typing import List
import types
import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize


# Recursive function to collect leaf layers and their names
def collect_leaf_layers(layer, prefix=""):
    leaf_layers = []  # List to store the leaf layers and their names

    for name, sub_layer in layer.named_children():
        full_name = prefix + "." + name if prefix else name

        # If the sub-layer is a leaf layer, append the leaf layer and its full name to the list. If not a leaf layer, recursively go deeper
        if len(list(sub_layer.children())) == 0:  # Leaf layer check
            leaf_layers.append((sub_layer, full_name))
        else:
            leaf_layers.extend(collect_leaf_layers(sub_layer, full_name))

    return leaf_layers  # Return the collected list of leaf layers


def scaling_layers(model, up_size: int = 1, layer_name: list = None):
    """
    Replace Conv2D layers with scaled kernels and optionally target specific layers.
    
    Parameters:
    - model (nn.Module): The model to modify.
    - up_size (int): The amount to increase the kernel size by on each side.
    - layer_name (list): Optional. List of specific full names of layers to replace.

    Returns:
    - model (nn.Module): The modified model.
    """
    # Collect all leaf layers
    leaf_layers = collect_leaf_layers(model)

    # Iterate through the leaf layers
    for layer, full_name in leaf_layers:
        if isinstance(layer, nn.modules.conv._ConvNd):
            # Skip the Conv layer with kernel_size=1 or not in layer_name if specified
            if (min(layer.kernel_size) == 1) or (layer_name and full_name not in layer_name):
                pass  # Skip scaling but freeze parameters below
            else:
                # Replace the original kernel_size and padding
                layer.kernel_size = tuple(k + 2 * up_size for k in layer.kernel_size)
                layer.padding = tuple(p + up_size for p in layer.padding)

                # Initialize the base weight with the original values at the center
                slices = tuple(slice(up_size, -up_size) for _ in range(layer.weight.ndim - 2))
                base_weight = torch.zeros(*[v + 2 * up_size if i > 1 else v for i, v in enumerate(layer.weight.shape)])
                base_weight[(..., *slices)] = layer.weight.data
                # constant_tensor = torch.tensor(constant_value, dtype=torch.float32)
                layer.register_buffer('base_weight', base_weight)
                # layer.base_weight = torch.zeros(*[v + 2 * up_size if i > 1 else v for i, v in enumerate(layer.weight.shape)])
                # layer.base_weight[(..., *slices)] = layer.weight.data

                # Replace the original weight with the new one
                layer.weight = nn.Parameter(torch.zeros(*[v + 2 * up_size if i > 1 else v for i, v in enumerate(layer.weight.shape)]))

                def forward(self, input: torch.Tensor) -> torch.Tensor:
                    return self._conv_forward(input, self.weight + self.base_weight, self.bias)

                layer.forward = types.MethodType(forward, layer)

                # Skip freeze parameters
                continue
        # Freeze parameters for all layers that were not modified
        for param in layer.parameters():
            param.requires_grad = False

    return model


class LinearTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Linear Transform Layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
    
    def forward(self, x):
        """
        Forward pass through the Linear Transform Layer
        :param x: Input tensor of shape [N, ..., in_channels]
        :return: Output tensor of shape [N, out_channels, ...]
        """
        x = self.linear(x)  # [N, ..., in_channels]
        output = x.permute(0, -1, *range(1, x.ndim - 1))  # [N, out_channels, ...]
        return output
    

class FourierTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_terms: int):
        """
        Fourier Transform Layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param num_terms: Number of Fourier terms
        """
        super(FourierTransform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_terms = num_terms

        # Learnable weights for cosine and sine terms
        self.cos_weight = nn.Parameter(torch.zeros(out_channels, in_channels * num_terms))
        self.sin_weight = nn.Parameter(torch.zeros(out_channels, in_channels * num_terms))

    def forward(self, x):
        """
        Forward pass through the Fourier Transform Layer
        :param x: Input tensor of shape [N, in_channels, ...]
        :return: Output tensor of shape [N, out_channels, ...]
        """
        # Input shape [N, in_channels, ...]
        N, in_channels, *spatial_dims = x.shape
        x_flat = x.view(N, in_channels, -1)  # Shape [N, in_channels, prod(...)]
        
        # Generate Fourier terms
        n = torch.arange(1, self.num_terms + 1, device=x.device).view(1, 1, -1)  # Shape [1, 1, n_terms]
        x_expanded = x_flat.unsqueeze(-1)  # Shape [N, in_channels, prod(...), 1]
        nx = n * x_expanded  # Shape [N, in_channels, prod(...), n_terms]

        # Compute cosine and sine terms
        cos_terms = torch.cos(nx).permute(0, 1, 3, 2).reshape(N, in_channels * self.num_terms, -1)  # Shape [N, in_channels*n_terms, prod(...)]
        sin_terms = torch.sin(nx).permute(0, 1, 3, 2).reshape(N, in_channels * self.num_terms, -1)  # Shape [N, in_channels*n_terms, prod(...)]

        # Compute cosine and sine contributions
        cos_part = torch.matmul(self.cos_weight, cos_terms)  # Shape [N, out_channels, prod(...)]
        sin_part = torch.matmul(self.sin_weight, sin_terms)  # Shape [N, out_channels, prod(...)]

        # Combine cosine and sine parts
        output = cos_part + sin_part  # Shape [N, out_channels, prod(...)]

        # Reshape back to original spatial dimensions
        output = output.view(N, self.out_channels, *spatial_dims)  # Shape [N, out_channels, ...]
        return output


class Permute(nn.Module):
    def __init__(self):
        """
        Linear Transform Layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super(Permute, self).__init__()
    
    def forward(self, x):
        """
        Forward pass through the Linear Transform Layer
        :param x: Input tensor of shape [N, out_channels, ...]
        :return: Output tensor of shape [N, ..., out_channels]
        """
        output = x.permute(0, *range(2, x.ndim), 1)
        return output
    

def layer_wrapper(layer: nn.Linear, rank=4, num_terms=8):
    device = next(layer.parameters()).device
    if isinstance(layer, nn.Linear):
        channels = layer.in_features
    elif isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv3d):
        channels = layer.in_channels
    else:
        raise ValueError("The class of parameter `layer` should be one of `nn.Linear`, `nn.Conv1D`, `nn.Conv2D`, and `nn.Conv3D`.")
    layer.remainder = nn.Sequential(
        LinearTransform(in_channels=channels, out_channels=rank),
        nn.ReLU(),
        FourierTransform(in_channels=rank, out_channels=channels, num_terms=num_terms), 
        Permute()
    ).to(device)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.__class__.forward(self, input + self.remainder(input))
    
    layer.forward = types.MethodType(forward, layer)

    return layer