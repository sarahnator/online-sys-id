import torch
from torch.utils.data import IterableDataset
import matplotlib.pyplot as plt

from architecture.neural_ode import rk4_step

def van_der_pol(t, x, mu=1.0):
    x1 = x[..., 0]
    x2 = x[..., 1]
    dx1 = x2
    dx2 = mu * (1 - x1 ** 2) * x2 - x1
    return torch.stack([dx1, dx2], dim=-1)

class VanDerPolDataset(IterableDataset):
    """
    Iterable dataset for the Van der Pol oscillator.
    Each sample corresponds to a single initial condition and time step, with a fixed mu parameter.
    The dataset generates a batch of initial conditions and time steps, integrates the Van der Pol oscillator for one step,
    and returns the initial conditions, time steps, and resulting states.
    """
    def __init__(
        self,
        n_points: int = 1000,
        n_example_points: int = 100,
        mu_range=(0.5, 2.5),
        y0_range=(-3.5, 3.5),
        # dt_range=(0.01, 0.1),
        dt_range=(0.1, 0.1),
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
            # Generate a single mu for all points in the batch
            # This simulates a scenario where the dynamics are constant but the initial conditions and time steps vary
            mu = torch.empty(1).uniform_(*self.mu_range)
            # Generate random initial conditions
            _y0 = torch.empty(total_points, 2).uniform_(*self.y0_range)
            # Generate random time steps - same dynamics mu but different time resolution to simulate realistic non-constant sampling intervals
            _dt = torch.empty(total_points).uniform_(*self.dt_range)
            # Integrate one step of the Van der Pol oscillator to isolate the instantaneous behavior for system identification or short-term prediction (and avoid accumulation of numerical errors)
            _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu)

            # Split the data
            y0_example = _y0[: self.n_example_points]
            dt_example = _dt[: self.n_example_points]
            y1_example = _y1[: self.n_example_points]

            y0 = _y0[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            y1 = _y1[self.n_example_points :]

            info = {
                'mu': mu,
            }
            yield y0, dt, y1, y0_example, dt_example, y1_example, info

    def plot_single_trajectory(self, mu: float, y0: torch.Tensor):
        """
        Plot a single trajectory of the Van der Pol oscillator dataset.
        This function integrates the Van der Pol oscillator for a given initial condition and plots the resulting trajectory.

        Args:
            mu (float): The mu parameter of the Van der Pol oscillator.
            y0 (torch.Tensor): Initial conditions of shape (1, 2).
        """
        import matplotlib.pyplot as plt

        # Integrate the Van der Pol oscillator for the given initial conditions and time steps
        trajectory = [y0.clone()]  # Initialize the first point (1, 2)
        s = 0.1  # Time step for simulation
        n = int(10 / s) # Number of steps to simulate
        _dt = torch.tensor([s], device=y0.device)

        for i in range(n):
            current_state = trajectory[-1]
            change_in_state = rk4_step(van_der_pol, current_state, _dt, mu=mu)
            resulting_state = current_state + change_in_state
            trajectory.append(resulting_state)
        trajectory = torch.stack(trajectory, dim=0)

        trajectory = trajectory.detach().cpu().numpy()

        # plot
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f"Van der Pol Oscillator", fontsize=16)
        plt.plot(trajectory[:, 0, 0], trajectory[:, 0, 1])
        plt.title(f"$\\mu$={mu.item():0.1f}")
        # label the initial condition and final point
        plt.scatter(
            trajectory[0, 0, 0], trajectory[0, 0, 1], color="blue", label="Initial Condition"
        )
        plt.scatter(
            trajectory[-1, 0, 0], trajectory[-1, 0, 1], color="red", label="Terminal State"
        )
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # def integrate_batch(self, y0s, dt, mu):
    #     """
    #     Integrate a batch of initial conditions and time steps using the Van der Pol oscillator dynamics.
        
    #     Args:
    #         y0s (torch.Tensor): Initial conditions of shape (batch_size, 2).
    #         dt (torch.Tensor): Time step of shape (batch_size,).
    #         mu (float): The mu parameter of the Van der Pol oscillator.
        
    #     Returns:
    #         torch.Tensor: Integrated states of shape (batch_size, 2).
    #     """
    #                 # Integrate the true trajectory
    #         x = _y0.clone()
    #         y = [x] # track all points in the trajectory
    #         for k in range(n):
    #             x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
    #             y.append(x)
    #         y = torch.cat(y, dim=0) # concatenate into a single tensor along the time dimension
    #         y = y.detach().cpu().numpy()
        



    # def plot_vander_pol(y0s, yhats, dt, mu):
    #     """
    #     Plot the Van der Pol oscillator trajectories for a batch of initial conditions and predicted states.
        
    #     Args:
    #         y0s (torch.Tensor): Initial conditions of shape (batch_size, 2).
    #         yhat (torch.Tensor): Predicted states of shape (batch_size, 2).
    #         dt (torch.Tensor): Time step of shape (batch_size,).
    #         mu (float): The mu parameter of the Van der Pol oscillator.
    #     """
    #     import matplotlib.pyplot as plt

    #     # Convert tensors to numpy for plotting
    #     y0s = y0s.detach().cpu().numpy()
    #     yhat = yhat.detach().cpu().numpy()

    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax.set_title(f"Van der Pol Oscillator Trajectories with $\\mu$={mu.item():.2f}")
    #     ax.set_xlim(-5, 5)
    #     ax.set_ylim(-5, 5)

    #     for i in range(len(y0s)):
    #         # Integrate the Van der Pol oscillator for each initial condition
    #         trajectory = [y0s[i]]
    #         n_steps = int(10 / dt[i].item())
    #         for _ in range(n_steps):
    #             current_state = trajectory[-1]
    #             change_in_state = rk4_step(van_der_pol, current_state, dt[i:i+1], mu=mu)
    #             resulting_state = current_state + change_in_state
    #             trajectory.append(resulting_state)
    #         trajectory = torch.stack(trajectory, dim=0).detach().cpu().numpy()

    #         # Plot the trajectory
    #         ax.plot(trajectory[:, 0], trajectory[:, 1], label=f"Trajectory {i+1}")

    #         # Plot the predicted state
    #         ax.scatter(yhat[i, 0], yhat[i, 1], color='red', marker='x', label=f"Predicted {i+1}")

    #     ax.set_xlabel("x1")
    #     ax.set_ylabel("x2")
    #     ax.legend()
    #     plt.show()

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # set the device to cuda if available, otherwise mps or cpu
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch.manual_seed(42)

    # generate a batch of data to plot as a test
    dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
    dataloader = DataLoader(dataset, batch_size=9)
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)

    y0, dt, y1, y0_example, dt_example, y1_example, info = batch
    mu = info['mu'].to(device)
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    # Plot a single trajectory from the dataset with the first initial condition from the first batch
    dataset.plot_single_trajectory(mu[0], y0_example[0,0].unsqueeze(0))

    # Plot first 9 trajectories from the dataset
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        for j in range(3):

            # Plot a single trajectory
            _mu = mu[i * 3 + j]
            _y0 = torch.empty(1, 2, device=device).uniform_(
                *dataloader.dataset.y0_range
            )

            s = 0.1  # Time step for simulation
            n = int(10 / s)
            _dt = torch.tensor([s], device=device)

            # Integrate the true trajectory
            x = _y0.clone()
            y = [x] # track all points in the trajectory
            for k in range(n):
                x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
                y.append(x)
            y = torch.cat(y, dim=0) # concatenate into a single tensor along the time dimension
            y = y.detach().cpu().numpy()

            ax[i, j].set_xlim(-5, 5)
            ax[i, j].set_ylim(-5, 5)
            (_t,) = ax[i, j].plot(y[:, 0], y[:, 1])
            ax[i, j].set_title(f"$\\mu$={_mu.item():.2f}")

    # Add a single legend for all subplots
    _t.set_label("Van der Pol Trajectory")

    fig.legend(
        handles=[_t],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )

    plt.show()

