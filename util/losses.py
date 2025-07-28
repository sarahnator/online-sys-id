import torch
from typing import Tuple
from util.coefficients import least_squares

def basis_normalization_loss(K: torch.Tensor) -> torch.Tensor:
    """Penalize the diagonal of the gram matrix being far from one.

    Args:
        K (torch.Tensor): Gram matrix [batch_size, n_basis, n_basis]

    Returns:
        torch.Tensor: Mean squared difference of diagonal elements from one
    """
    return ((torch.diagonal(K, dim1=-2, dim2=-1) - 1) ** 2).mean()


def basis_orthonormality_loss(K: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Penalize the gram matrix being far from the identity.

    Args:
        K (torch.Tensor): Gram matrix [batch_size, n_basis, n_basis]
        device (torch.device): Device on which to create the identity matrix

    Returns:
        torch.Tensor: Mean norm of the difference between K and the identity matrix
    """
    identity_matrix = torch.eye(K.shape[-1], device=device)
    gram_matrix_penalty = (K - identity_matrix).norm(dim=(1, 2)).mean()
    return gram_matrix_penalty


def residual_loss(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Compute the mean squared error loss between the residual prediction and targets.

    Args:
        model (torch.nn.Module): Model with a residual_function
        inputs (torch.Tensor): Input tensor [batch_size, n_points, n_features]
        targets (torch.Tensor): Target tensor [batch_size, n_points, n_features]

    Returns:
        torch.Tensor: Mean squared error loss
    """
    residual_pred = model.residual_function(inputs)
    return torch.nn.functional.mse_loss(residual_pred, targets)

def neural_ode_loss(
    model: torch.nn.Module, batch: Tuple[
    torch.Tensor,  # y0
    torch.Tensor,  # dt
    torch.Tensor,  # y1
    torch.Tensor,  # y0_example
    torch.Tensor,  # dt_example
    torch.Tensor,  # y1_example
    dict,  # info
], device: torch.device) -> torch.Tensor:
    """Compute the neural ODE loss.

    Args:
        model (torch.nn.Module): Model with a residual_function
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]): A batch containing:
            - y0: Initial conditions
            - dt: Time steps
            - y1: Target values after time step
            - y0_example: Example initial conditions
            - dt_example: Example time steps
            - y1_example: Example target values after time step
            - info: Additional information (e.g., parameters)
    Returns:
        torch.Tensor: Neural ODE loss
    """
    y0, dt, y1, y0_example, dt_example, y1_example, info = batch

    # move to device
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    try: # Function Encoder
        coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)
        prediction = model((y0, dt), coefficients=coefficients)
    except Exception as e:
        # print(f"Error in computing coefficients: {e}")
        # prediction = model((y0, dt), y1)
        try: # Pure Neural ODE
            prediction = model((y0, dt))
        except Exception as e: # MAML: we have to call the inner model which is a Neural ODE
            prediction = model.model((y0, dt))
            # prediction = model((y0, dt), {}) # also works?

    prediction_loss = torch.nn.functional.mse_loss(prediction, y1)
    return prediction_loss

def mlp_loss(model: torch.nn.Module, batch: Tuple[
    torch.Tensor,  # y0
    torch.Tensor,  # y1
    torch.Tensor,  # y0_example
    torch.Tensor,  # y1_example
    dict,  # info
], device: torch.device) -> torch.Tensor:
    """Compute the MLP loss.

    Args:
        model (torch.nn.Module): Model with a residual_function
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]): A batch containing:
            - y0: Initial conditions
            - dt: Time steps
            - y1: Target values after time step
            - y0_example: Example initial conditions
            - dt_example: Example time steps
            - y1_example: Example target values after time step
            - info: Additional information (e.g., parameters)
    Returns:
        torch.Tensor: Neural ODE loss
    """
    y0, dt, y1, y0_example, dt_example, y1_example, info = batch

    # move to device
    y0 = y0.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    y1_example = y1_example.to(device)

    try:
        coefficients, G = model.compute_coefficients(y0_example, y1_example)
        prediction = model(y0, coefficients=coefficients)
    except Exception as e:
        # print(f"Error in computing coefficients: {e}")
        prediction = model(y0)

    prediction_loss = torch.nn.functional.mse_loss(prediction, y1)
    return prediction_loss