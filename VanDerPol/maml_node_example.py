from typing import Callable, Optional, Tuple, Union
from util.losses import neural_ode_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import *

from models.get_model import get_model
from models.maml import *
import matplotlib.pyplot as plt
import time 

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
alg = 'MAML_NODE'
# alg = 'MAML2_NODE'
model = get_model(algorithm=alg, device=device)
model.load_state_dict(torch.load(f"./logs/VanDerPol_{alg}/model.pth", map_location=device))
model.loss_function = torch.nn.MSELoss()
model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


"""
Qualitative evaluation of the MAML model.
"""
def qualitative_evaluation():
    # make predictions with the model
    # fine-tune on examples
    batch = next(dataloader_iter)
    y0, dt, y1, y0_example, dt_example, y1_example, info = batch
    y0=y0.to(device)
    y1=y1.to(device)
    dt=dt.to(device)
    dt_example = dt_example.to(device)
    y0_example = y0_example.to(device)
    y1_example = y1_example.to(device)
    adapted_weights, _ = model.inner_update_step(x=y0_example, dt=dt_example, y=y1_example) # note here we use the last parameter estimate 

    # Roll out the model on the batch of initial conditions
    # and compare to the ground truth trajectory
    s = 0.1 # simulation time step
    n = int(10 / s)  # number of steps to simulate

    ground_truth_traj = [y0.clone()]
    maml_traj = [y0.clone()]
    _dt = torch.tensor([s], device=device)  # time step for simulation
    for i in range(n):
        # ground truth trajectory
        current_state = ground_truth_traj[-1]
        change_in_state = rk4_step(van_der_pol, current_state, _dt, mu=info['mu'].to(device))
        resulting_state = current_state + change_in_state
        ground_truth_traj.append(resulting_state)

        maml_state = maml_traj[-1]
        # expand _dt to match the batch size and number of points
        expanded_dt = _dt.expand(maml_state.shape[0], -1).expand(-1, maml_state.shape[1])
        maml_dstate = model.forward(inputs=(maml_state, expanded_dt), model_kwargs={'params': adapted_weights})
        maml_prediction = maml_state + maml_dstate
        maml_traj.append(maml_prediction)

    ground_truth_traj = torch.stack(ground_truth_traj, dim=0).detach().cpu().numpy()
    maml_traj = torch.stack(maml_traj, dim=0).detach().cpu().numpy()

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
            (_m,) = ax[i, j].plot(maml_traj[:, traj_idx, 0, 0], maml_traj[:, traj_idx, 0, 1], color='orange', label='MAML')

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

def quantitative_evaluation(mu_function: Callable = mu_piecewise_constant, trange: int = 5000, n_trials: int = 1):
    # mu = torch.empty(1, device=device).uniform_(*dataset.mu_range) # random initial mu parameter
    # plotting_mu = [mu.item()]  # for plotting purposes, we will keep track of the mu parameter
    mu = mu_function(t=torch.arange(trange, device=device), device=device) 
    mu = torch.cat([mu] * n_trials, dim=-1)
    # time-varying mu parameter
    losses_maml = []  # to store the losses for each step
    adapted_weights = copy_model_params(model.model, 1)  # copy the parameters for each task in the batch, this is a placeholder for the first step

    cumulative_data = { 'x': [], 'y': [], 'dt': [] }  # to store the data for each step
    window_size = getattr(model, "window_size", 1)  # number of observations to keep in the window for online adaptation

    compute_time = []  # to store the time it takes to adapt the model and make predictions
    with tqdm.trange(trange * n_trials) as tqdm_bar:
        for step in tqdm_bar:

            # # Update the mu parameter every 500 steps
            # if step % 1000 == 0 and step > 0:
            #     mu = torch.empty(1, device=device).uniform_(*dataset.mu_range)
            #     plotting_mu.append(mu.item())
            mu_t = mu[step]

            # Generate a new observation
            y0 = torch.empty(1, 1, 2, device=device).uniform_(*dataset.y0_range)
            dt = torch.empty(1, 1, device=device).uniform_(*dataset.dt_range)
            y1 = rk4_step(van_der_pol, y0, dt, mu=mu_t)

            if window_size > 1:
                # add to cumulative data
                cumulative_data['x'].append(y0)
                cumulative_data['y'].append(y1)
                cumulative_data['dt'].append(dt)
                for key in cumulative_data:
                    cumulative_data[key] = cumulative_data[key][-window_size:] # keep only the last window_size observations

                tensorized_data = {key: torch.cat(value, dim=1) for key, value in cumulative_data.items()}
            else:
                tensorized_data = {
                    'x': y0,
                    'y': y1,
                    'dt': dt
                }

            # Generate a new batch of data for evaluation
            n_points = 1000
            _y0 = torch.empty(1, n_points, 2, device=device).uniform_(*dataset.y0_range)
            _dt = torch.empty(1, n_points, device=device).uniform_(*dataset.dt_range)
            _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu_t)

            # Adapt the model using the new observation and predict the next step
            # track the time it takes to adapt the model
            t0 = time.perf_counter()

            adapted_weights, _ = model.inner_update_step_from_params(x=tensorized_data['x'], dt=tensorized_data['dt'], y=tensorized_data['y'], params=adapted_weights)

            # Compute maml predictions with update on the single observation
            # pred, loss, adapted_weights = model.datastream_predict(xs=y0, dt=dt, ys=y1, query_xs=_y0, query_dt=_dt, query_ys=_y1)
            maml_pred = model.forward((_y0, _dt), model_kwargs={'params': adapted_weights})
            t1 = time.perf_counter()
            compute_time.append(t1 - t0)
            loss = model.loss_function(maml_pred, _y1)
            
            losses_maml.append(loss.item())

            tqdm_bar.set_postfix(
                {
                    "loss_maml": f"{loss.item():.2e}",
                }
            )
    
    # Plot the loss
    mu_func_string = mu_function.__name__
    fig, ax = plt.subplots(1, 1, figsize=(4*10,10))

    # plot mu as vertical lines for every 1000 steps
    plotting_mu = mu[torch.arange(trange * n_trials) % 1000 == 0].detach().cpu().numpy().tolist()

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
    ax.plot(losses_maml, label="MAML NODE Loss", color='blue')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"./logs/VanDerPol_{alg}/losses_{mu_func_string}.png", bbox_inches='tight')

    # save the losses
    losses_maml = torch.tensor(losses_maml, device=device)
    torch.save({
        "losses_maml": losses_maml,
        "compute_time": compute_time,
        "mu": mu,
    }, f"./logs/VanDerPol_{alg}/losses_{mu_func_string}.pth")


if __name__ == "__main__":
    # qualitative_evaluation()
    quantitative_evaluation(mu_function=mu_piecewise_constant, trange=5000, n_trials=20)
    # quantitative_evaluation(mu_function=mu_sinusoidal_modulation, trange=5000)
    # quantitative_evaluation(mu_function=mu_linear_ramp, trange=5000)
    # quantitative_evaluation(mu_function=mu_constant, trange=5000)
    print("Evaluation completed.")
