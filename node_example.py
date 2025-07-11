from typing import Callable, Optional, Tuple, Union
from models.maml import copy_model_params
from util.online_adapt import online_adapt_maml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from architecture.neural_ode import rk4_step

from datasets.van_der_pol import VanDerPolDataset, van_der_pol, mu_piecewise_constant, mu_constant, mu_linear_ramp, mu_sinusoidal_modulation

from models.get_model import get_model
import matplotlib.pyplot as plt

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

# Load dataset
dataset = VanDerPolDataset(n_points=100, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
alg = 'NODE'
model = get_model(algorithm=alg, device=device)
model.load_state_dict(torch.load(f"./logs/VanDerPol_{alg}/model.pth", map_location=device))
model.loss_function = torch.nn.MSELoss()
model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def qualitative_evaluation():
    """
    Qualitative evaluation.
    """

    batch = next(dataloader_iter)
    y0, dt, y1, y0_example, dt_example, y1_example, info = batch
    y0=y0.to(device)
    y1=y1.to(device)
    dt=dt.to(device)
    dt_example = dt_example.to(device)
    y0_example = y0_example.to(device)
    y1_example = y1_example.to(device)

    # Roll out the model on the batch of initial conditions
    # and compare to the ground truth trajectory
    s = 0.1 # simulation time step
    n = int(10 / s)  # number of steps to simulate

    ground_truth_traj = [y0.clone()]
    node_traj = [y0.clone()]
    _dt = torch.tensor([s], device=device)  # time step for simulation
    for i in range(n):
        # ground truth trajectory
        current_state = ground_truth_traj[-1]
        change_in_state = rk4_step(van_der_pol, current_state, _dt, mu=info['mu'].to(device))
        resulting_state = current_state + change_in_state
        ground_truth_traj.append(resulting_state)

        node_state = node_traj[-1]
        # expand _dt to match the batch size and number of points
        expanded_dt = _dt.expand(node_state.shape[0], -1).expand(-1, node_state.shape[1])
        node_dstate = model.forward(inputs=(node_state, expanded_dt))
        node_prediction = node_state + node_dstate
        node_traj.append(node_prediction)


    ground_truth_traj = torch.stack(ground_truth_traj, dim=0).detach().cpu().numpy()
    node_traj = torch.stack(node_traj, dim=0).detach().cpu().numpy()

    # Plot first 9 trajectories from the dataset
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    mus = info['mu'].detach().cpu().numpy()
    for i in range(3):
        for j in range(3):
            traj_idx = i * 3 + j
            _mu = mus[i * 3 + j]
            ax[i, j].set_title(f"$\\mu$={_mu.item():.1f}")
            ax[i, j].set_xlim(-5, 5)
            ax[i, j].set_ylim(-5, 5)
            (_t,) = ax[i, j].plot(ground_truth_traj[:, traj_idx, 0, 0], ground_truth_traj[:, traj_idx, 0, 1], color='blue', label='True')
            (_m,) = ax[i, j].plot(node_traj[:, traj_idx, 0, 0], node_traj[:, traj_idx, 0, 1], color='orange', label='MAML')

    fig.legend(
        handles=[_t, _m],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )

    # plt.show()
    # save the figure
    fig.savefig(f"./logs/VanDerPol_{alg}/qualitative_example.png", bbox_inches='tight')
    plt.close(fig)


def quantitative_evaluation(mu_function: Callable = mu_piecewise_constant, trange: int = 5000):
    """
    This function performs a quantitative evaluation of the model by computing
    the mean squared error (MSE) between the model predictions and the ground truth.
    
    Args:
        mu_function (callable): A function that takes a time tensor and returns a time-varying mu tensor.
        trange (int): The range of time steps for evaluation.
    """
    mu = mu_function(t=torch.arange(trange, device=device), device=device)  # time-varying mu parameter
    # mu = torch.empty(1, device=device).uniform_(*dataset.mu_range) # random initial mu parameter
    losses_node = []  # to store the losses for each step
    losses_node_cumulative = []  # to store the losses of the cumulative version of maml for each step
    losses_node_window  = []  # to store the losses of the windowed version of maml for each step

    cumulative_data = []
    # set the update MAML parameters
    lr = 1e-3
    window = 100  # use a window of 100 steps for the windowed version of maml

    _adapted_params = copy_model_params(model, 1)  # copy the parameters for each task in the batch, this is a placeholder for the first step
    _adapted_windowed_params = _adapted_params.copy()  # copy the parameters for the windowed version of maml
    with tqdm.trange(trange) as tqdm_bar:
        for step in tqdm_bar:

            # # Update the mu parameter every 500 steps
            # if step % change_mu_every == 0 and step > 0:
            #     mu = torch.empty(1, device=device).uniform_(*dataset.mu_range)
            #     plotting_mu.append(mu.item())
            mu_t = mu[step]

            # Generate a new observation
            y0 = torch.empty(1, 1, 2, device=device).uniform_(*dataset.y0_range)
            dt = torch.empty(1, 1, device=device).uniform_(*dataset.dt_range)
            y1 = rk4_step(van_der_pol, y0, dt, mu=mu_t)
            
            # Append the observation to the cumulative data
            cumulative_data.append(((y0, dt), y1))
            
            adapted_params, loss, _ = online_adapt_maml(
                model=model,
                loss_fn=model.loss_function,
                data_stream=[((y0, dt), y1)],  # single observation
                lr=lr,
                use_full_history=False,  # we only use the single observation
                _params=_adapted_params,  # use the current adapted parameters
            )

            # # Cumulative version of maml
            # cumulative_adapted_params, cumulative_loss, _ = online_adapt_maml(
            #     model=model,
            #     loss_fn=model.loss_function,
            #     data_stream=cumulative_data,  # all observations so far
            #     lr=lr,
            #     use_full_history=True,  # we use all observations
            #     _params=_adapted_params,  # use the current adapted parameters
            # )

            # Windowed version of maml
            windowed_data = cumulative_data[-window:] if len(cumulative_data) > window else cumulative_data
            windowed_adapted_params, windowed_loss, _ = online_adapt_maml(
                model=model,
                loss_fn=model.loss_function,
                data_stream=windowed_data,  # last `window` observations
                lr=lr,
                use_full_history=True,  # we use all observations in the window
                _params=_adapted_windowed_params,  # use the current adapted windowed parameters
            )

            # Generate a new batch of data for evaluation
            n_points = 1000
            _y0 = torch.empty(1, n_points, 2, device=device).uniform_(*dataset.y0_range)
            _dt = torch.empty(1, n_points, device=device).uniform_(*dataset.dt_range)
            _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu_t)

            # Compute node predictions with update on the single observation
            node_pred = model.forward((_y0, _dt), {'params': adapted_params})
            loss = model.loss_function(node_pred, _y1)

            ## Compute node predictions with cumulative update
            # cumulative_node_pred = model.forward((_y0, _dt), {'params': cumulative_adapted_params})
            # cumulative_loss = model.loss_function(cumulative_node_pred, _y1)

            # Compute node predictions with windowed update
            windowed_node_pred = model.forward((_y0, _dt), {'params': windowed_adapted_params})
            windowed_loss = model.loss_function(windowed_node_pred, _y1)
            
            # Store the losses
            losses_node.append(loss.item())
            # losses_node_cumulative.append(cumulative_loss.item())
            losses_node_window.append(windowed_loss.item())


            tqdm_bar.set_postfix(
                {
                    "loss_node": f"{loss.item():.2e}",
                    # "loss_node_cumulative": f"{cumulative_loss.item():.2e}",
                    "loss_node_window": f"{windowed_loss.item():.2e}",
                }
            )

            # set the adapted parameters for the next step
            _adapted_params = adapted_params.copy()
            _adapted_windowed_params = windowed_adapted_params.copy()

    mu_func_string = mu_function.__name__

    # Plot the stand alone loss
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # plot mu as vertical lines for every 1000 steps
    plotting_mu = mu[torch.arange(trange) % 1000 == 0].detach().cpu().numpy().tolist()
    for i, m in enumerate(plotting_mu):
        x = i * 1000
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        ax.text(
            x,               # data x
            0.1,               # axis-fraction y = 0 (bottom of the plotting area)
            f"$\\mu$={m:.2f}",
            transform=ax.get_xaxis_transform(),  # <-- key!
            rotation=90,
            va='bottom',     # push the text upward from the axis spine
            ha='left',
            fontsize=11,
            fontweight='bold',
        )
        
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(which="both", axis="y", linestyle=":", linewidth=0.5)
    ax.plot(losses_node, label="MAML NODE Loss", color='blue')
    # ax.plot(losses_node_cumulative, label="MAML (cumulative data) NODE Loss", color='orange')
    ax.plot(losses_node_window, label="MAML (windowed data) NODE Loss", linestyle='--', color='green')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"./logs/VanDerPol_{alg}/losses_{mu_func_string}_{window}.png", bbox_inches='tight')

    # save the losses
    losses_node = torch.tensor(losses_node, device=device)
    losses_node_cumulative = torch.tensor(losses_node_cumulative, device=device)
    torch.save({
        "mu": mu,
        "losses_node": losses_node,
        # "losses_node_cumulative": losses_node_cumulative,
        "losses_node_window": torch.tensor(losses_node_window, device=device),
    }, f"./logs/VanDerPol_{alg}/losses_{mu_func_string}_{window}.pth")


if __name__ == "__main__":
    qualitative_evaluation()
    quantitative_evaluation(mu_function=mu_piecewise_constant, trange=5_000)
    quantitative_evaluation(mu_function=mu_sinusoidal_modulation, trange=5_000)
    quantitative_evaluation(mu_function=mu_linear_ramp, trange=5_000)
    quantitative_evaluation(mu_function=mu_constant, trange=5_000)
    print("Evaluation completed.")
