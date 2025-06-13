from models.function_encoder import FunctionEncoder, BasisFunctions
from models.maml import MAML
from architecture.mlp import MLP
from architecture.neural_ode import NeuralODE, ODEFunc, rk4_step
import torch
from util.losses import *

def get_loss_function(algorithm: str):
    """Get the loss function based on the specified algorithm.

    Args:
        algorithm (str): Type of the model ('NODE_FE', 'FE', 'MLP', 'NODE').
    Returns:
        Callable: The loss function to be used.
    """
    if algorithm == 'NODE_FE':
        return neural_ode_loss
    elif algorithm == 'MAML':
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown model type: {algorithm}")

def get_model(algorithm: str, n_layers:int, n_params: int, n_basis: int, device: str) -> torch.nn.Module:
    """Get a model based on the specified type.

    Args:
        algorithm (str): Type of the model ('NODE_FE', 'FE', 'MLP', 'NODE', 'MAML').
        n_params (int): Number of parameters for the model.
        n_basis (int): Number of basis functions for function encoders.
        device (str): Device to place the model on ('cpu' or 'cuda').
    Returns:
        torch.nn.Module: The initialized model.
    """


    if algorithm == 'NODE_FE':
        layer_sizes = [3, 64, 64, 2]  # Example layer sizes for MLP and NODE, adjust this later based on n_params
        activation = torch.nn.ReLU()
        bias = True

        # Create a FunctionEncoder with NeuralODE basis functions
        # can add residual function, but need to fix the rls prediction first.
        basis_functions = BasisFunctions(
            *[
                NeuralODE(
                    ode_func=ODEFunc(model=MLP(layer_sizes=layer_sizes, activation=activation, bias=bias)),
                    integrator=rk4_step,
                )
                for _ in range(n_basis)
            ]
        )
        return FunctionEncoder(
            basis_functions=basis_functions,
            # residual_function=MLP(layer_sizes=layer_sizes, activation=activation, bias=bias),
        ).to(device)

    # elif algorithm == 'FE':
    #     basis_functions = BasisFunctions(
    #         *[
    #             MLP(layer_sizes=layer_sizes, activation=activation, bias=bias)
    #             for _ in range(n_basis)
    #         ]
    #     )
    #     return FunctionEncoder(
    #         basis_functions=basis_functions,
    #         residual_function=MLP(layer_sizes=layer_sizes, activation=activation, bias=bias),
    #     ).to(device)

    # elif algorithm == 'MLP':
    #     return MLP(layer_sizes=layer_sizes, activation=activation, bias=bias)
    # elif algorithm == 'NODE':
    #     return NeuralODE(layer_sizes, activation=activation, bias=bias)
    elif algorithm == 'MAML':
        layer_sizes = [2, 64, 64, 2]  # Example layer sizes for MLP and NODE, adjust this later based on n_params
        activation = torch.nn.ReLU()
        bias = True
        return MAML(
            model=MLP(layer_sizes=layer_sizes, activation=activation, bias=bias),
            meta_learning_rate=1e-3,
            internal_learning_rate=1e-3,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {algorithm}")


