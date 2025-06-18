from typing import List, Callable, Union, Tuple, Mapping
import torch

class MLP(torch.nn.Module):
    """A simple multi-layer perceptron neural network.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.ReLU().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: Union[torch.nn.Module, Callable] = torch.nn.ReLU(),
        bias: bool = True,
    ):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        # Ensure activation is an nn.Module or callable
        self.activation = activation if isinstance(activation, torch.nn.Module) else activation
        
        # Build linear layers
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(in_f, out_f, bias=bias)
            for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor [batch_size, ...]

        Returns:
            torch.Tensor: Output tensor [batch_size, ...]
        """
        for layer in self.layers[:-1]:  # Apply all but last layer with activation
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)   # last layer without activation
        return x

    def forward_with_params(self, x: torch.tensor, params: List[Tuple[torch.tensor, torch.tensor]]) -> torch.tensor:
        """Forward pass through the MLP with custom weights.

        Args:
            x (torch.Tensor): Input tensor [batch_size, ...]
            params (List[Tuple[torch.Tensor, torch.Tensor]]): Custom weights and biases for the linear layers, of size batch_size x (n_layers, 2), where each tuple contains (weight, bias).

        Returns:
            torch.Tensor: Output tensor [batch_size, ...]
        """
        for i, layer in enumerate(self.layers[:-1]):
            W, b = params[i]
            # W has shape [batch_size, out_features, in_features]
            # b has shape [batch_size, out_features]
            # x has shape [batch_size, n_points, in_features]
    
            x = torch.einsum("bmn,bdn->bdm", W, x) + b.unsqueeze(1) # Ax+b , the bias is broad cast among the n_points dimension
            x = self.activation(x) # apply activation function
            # x = torch.nn.functional.linear(x, params[i][0], params[i][1])
            # torch.nn.functional.linear does not support batch-wise parameters, so compute manually

        # Apply the last layer without activation
        W, b = params[-1]
        x = torch.einsum("bmn,bdn->bdm", W, x) + b.unsqueeze(1)
        return x

