import torch
from torch.utils.data import IterableDataset
import matplotlib.pyplot as plt

def rk4_step(func, x, dt, **ode_kwargs):
    """Runge-Kutta 4th order ODE integrator for a single step."""
    t = torch.zeros_like(dt, device=dt.device)
    k1 = func(t, x, **ode_kwargs)
    k2 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k1, **ode_kwargs)
    k3 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k2, **ode_kwargs)
    k4 = func(t + dt, x + dt.unsqueeze(-1) * k3, **ode_kwargs)
    return (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)


def van_der_pol(t, x, mu=1.0):
    return torch.stack(
        [x[..., 1], mu * (1 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]], dim=-1
    )

class VanDerPolDataset(IterableDataset):
    def __init__(
        self,
        n_points: int = 1000,
        n_example_points: int = 100,
        mu_range=(0.5, 2.5),
        y0_range=(-3.5, 3.5),
        dt_range=(0.01, 0.1),
    ):
        super().__init__()
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.mu_range = mu_range
        self.y0_range = y0_range
        self.dt_range = dt_range

    def __iter__(self):
        while True:
            total_points = self.n_example_points + self.n_points
            # Generate a single mu
            mu = torch.empty(1).uniform_(*self.mu_range)
            # Generate random initial conditions
            _y0 = torch.empty(total_points, 2).uniform_(*self.y0_range)
            # Generate random time steps
            _dt = torch.empty(total_points).uniform_(*self.dt_range)
            # Integrate one step
            _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu)

            # Split the data
            y0_example = _y0[: self.n_example_points]
            dt_example = _dt[: self.n_example_points]
            y1_example = _y1[: self.n_example_points]

            y0 = _y0[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            y1 = _y1[self.n_example_points :]

            yield mu, y0, dt, y1, y0_example, dt_example, y1_example

    def plot(self, mu, y0, dt, y1, y0_example, dt_example, y1_example):

        plt.figure(figsize=(10, 5))
        plt.plot(y0[:, 0].numpy(), y0[:, 1].numpy(), 'o', label='Initial Conditions')
        plt.plot(y1[:, 0].numpy(), y1[:, 1].numpy(), 'x', label='Final Conditions')
        plt.plot(y0_example[:, 0].numpy(), y0_example[:, 1].numpy(), 'o', label='Example Initial Conditions')
        plt.plot(y1_example[:, 0].numpy(), y1_example[:, 1].numpy(), 'x', label='Example Final Conditions')
        plt.title(f'Van der Pol Oscillator (mu={mu.item():.2f})')
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.legend()
        plt.grid()
        plt.show()