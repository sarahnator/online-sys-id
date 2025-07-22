from typing import Callable, Optional, Tuple, Dict
import torch
from models.BaseModel import BaseModel

def rk4_step(func, x, dt, **ode_kwargs):
    """Runge-Kutta 4th order ODE integrator for a single step."""
    try:
        t = torch.zeros_like(dt, device=dt.device)
        k1 = func(t, x, **ode_kwargs)
        k2 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k1, **ode_kwargs)
        k3 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k2, **ode_kwargs)
        k4 = func(t + dt, x + dt.unsqueeze(-1) * k3, **ode_kwargs)
    except Exception as e:
        # shape mismatch or other error
        t = torch.zeros_like(dt, device=dt.device)
        u = x[..., -2:]  # Assuming the last two dimensions are control inputs

        k1 = func(t, x, **ode_kwargs)

        _k1 = torch.cat([x[..., :-2] + (dt / 2).unsqueeze(-1) * k1, u], dim=-1)
        k2 = func(t + dt / 2, _k1, **ode_kwargs)

        _k2 = torch.cat([x[..., :-2] + (dt / 2).unsqueeze(-1) * k2, u], dim=-1)
        k3 = func(t + dt / 2, _k2, **ode_kwargs)

        _k3 = torch.cat([x[..., :-2] + dt.unsqueeze(-1) * k3, u], dim=-1)
        k4 = func(t + dt, _k3, **ode_kwargs)
        
    return (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)


class ODEFunc(torch.nn.Module):
    """A wrapper for a PyTorch model to make it compatible with ODE solvers.

    Args:
        model (torch.nn.Module): The neural network model.
    """

    def __init__(self, model: torch.nn.Module):
        super(ODEFunc, self).__init__()
        self.model = model

    def forward(self, t, x, params=None):
        """Compute the time derivative at the current state.

        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state
            

        Returns:
            torch.Tensor: The time derivative dx/dt at the current state
        """
        tx = torch.cat([t.unsqueeze(-1), x], dim=-1)  # Concatenate time and state
        return self.model(tx, params)
        # if params is None:
        #     return self.model(tx)
        # else:
        #     return self.model.forward_with_params(tx, params)

class NeuralODE(BaseModel):
# class NeuralODE(torch.nn.Module):
    """Neural Ordinary Differential Equation model.

    Args:
        ode_func (torch.nn.Module): The vector field
        integrator (Callable): The ODE solver (e.g., `rk4_step`, `odeint`).
    """

    def __init__(
        self,
        ode_func: Callable,
        integrator: Callable,
    ):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.integrator = integrator

    @property
    def layers(self):
        return self.ode_func.model.layers

    def forward(
        self,
        inputs,
        ode_kwargs: Optional[Dict] = {},
    ):
        """Solve the initial value problem.

        Args:
            inputs (tuple): A tuple containing (y0, t), where:
                y0 (torch.Tensor): Initial condition
                dt (torch.Tensor): Time step
            ode_kwargs (dict, optional): Additional integrator arguments. Defaults to {}.

        Returns:
            torch.Tensor: Solution of the ODE at the next time step.
        """
        return self.integrator(self.ode_func, *inputs, **ode_kwargs)