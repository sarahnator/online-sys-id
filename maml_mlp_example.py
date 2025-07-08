from typing import Callable, Optional, Tuple, Union
from models.maml import copy_params

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import *

from models.get_model import get_model
from models.maml import *
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
# alg = 'MAML2_MLP'
alg = 'MAML_MLP'
model = get_model(algorithm=alg, device=device)
model.load_state_dict(torch.load(f"./logs/VanDerPol_{alg}/model.pth", map_location=device))
model.loss_function = torch.nn.MSELoss()
model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


"""
Qualitative evaluation of the MAML model on a single example from the dataset.
"""

# make predictions with the model
# fine-tune on examples
batch = next(dataloader_iter)
y0, dt, y1, y0_example, dt_example, y1_example, info = batch
y0=y0.to(device)
y1=y1.to(device)
dt=dt.to(device)
y0_example = y0_example.to(device)
y1_example = y1_example.to(device)
adapted_weights, _ = model.inner_update_step(x=y0_example, y=y1_example) # note here we use the last parameter estimate generated from a growing dataset

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
    maml_dstate = model.forward(x=maml_state, w=adapted_weights)
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


"""
Quantitative evaluation of the MAML model.
"""
mu = torch.empty(1, device=device).uniform_(*dataset.mu_range) # random initial mu parameter
plotting_mu = [mu.item()]  # for plotting purposes, we will keep track of the mu parameter
losses_maml_with_meta_update = []
losses_maml = []  # to store the losses for each step
adapted_weights = copy_params(model.model, 1)  # copy the parameters for each task in the batch, this is a placeholder for the first step
with tqdm.trange(5000) as tqdm_bar:
    for step in tqdm_bar:

        # Update the mu parameter every 500 steps
        if step % 1000 == 0 and step > 0:
            mu = torch.empty(1, device=device).uniform_(*dataset.mu_range)
            plotting_mu.append(mu.item())

        # Generate a new observation
        y0 = torch.empty(1, 1, 2, device=device).uniform_(*dataset.y0_range)
        dt = torch.empty(1, 1, device=device).uniform_(*dataset.dt_range)
        y1 = rk4_step(van_der_pol, y0, dt, mu=mu)

        # Compute the parameter estimate from data
        # TODO: Not sure which method to use here, whether to perform a meta-update or not.
        adapted_weights_maml_with_meta_update, _ = model.inner_update_step(x=y0, y=y1)
        model.meta_update_step(
            x=y0, y=y1, adapted_weights=adapted_weights_maml_with_meta_update, clip_grad_norm_=True  # take a meta step in the direction of the adapted weights
        )

        adapted_weights, _ = model.inner_update_step_from_params(x=y0, y=y1, params=adapted_weights)

        # Generate a new batch of data for evaluation
        n_points = 1000
        _y0 = torch.empty(1, n_points, 2, device=device).uniform_(*dataset.y0_range)
        _dt = torch.empty(1, n_points, device=device).uniform_(*dataset.dt_range)
        _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu)

        # Compute maml predictions
        maml_pred = model.forward(x=_y0, w=adapted_weights)
        loss_maml = torch.nn.functional.mse_loss(maml_pred, _y1)
        losses_maml.append(loss_maml.item())

        maml_pred_with_meta_update = model.forward(x=_y0, w=adapted_weights_maml_with_meta_update)
        loss_maml_with_meta_update = torch.nn.functional.mse_loss(maml_pred_with_meta_update, _y1)
        losses_maml_with_meta_update.append(loss_maml_with_meta_update.item())


        tqdm_bar.set_postfix(
            {
                "loss_maml": f"{loss_maml.item():.2e}",
                "loss_maml_with_meta_update": f"{loss_maml_with_meta_update.item():.2e}",
            }
        )

# Plot the loss
fig, ax = plt.subplots(1, 1, figsize=(10,10))
# plot a vertical dashed line at every 1000 steps and label with the mu parameter
for i in range(1, 6):
    ax.axvline(x=(i - 1) * 1000, color='gray', linestyle='--', linewidth=0.5)
    # label the line with the mu parameter
    ax.text(
        (i - 1) * 1000,
        1e-3,
        f"$\\mu$={plotting_mu[i-1]:.1f}",
        rotation=90,
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=11,
        # bold
        fontweight='bold',
    
    )

ax.set_yscale("log")
ax.plot(losses_maml, label="MAML Loss", color='orange')
ax.plot(losses_maml_with_meta_update, label="MAML Loss (with meta-update)", color='blue')
plt.legend()
plt.tight_layout()
# plt.show()
fig.savefig(f"./logs/VanDerPol_{alg}/losses.png", bbox_inches='tight')

# save the losses
losses_maml = torch.tensor(losses_maml, device=device)
losses_maml_with_meta_update = torch.tensor(losses_maml_with_meta_update, device=device)
torch.save({
    "losses_maml": losses_maml,
    "losses_maml_with_meta_update": losses_maml_with_meta_update,
}, f"./logs/VanDerPol_{alg}/losses.pth")