from typing import Callable, Optional, Tuple, Union
from models.maml import copy_model_params
from util.online_adapt import online_adapt_maml

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
# as you increase n_example_points, the model improves its predictions (see how the qualitative evaluation figure changes for the first mu!)
dataset = VanDerPolDataset(n_points=100, n_example_points=10_000, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
alg = 'MAML2_MLP'
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
params = copy_model_params(model.model, 1)  # copy the parameters for each task in the batch, this is a placeholder for the first step
adapted_weights, _ = model.inner_update_step(x=y0_example[0,...].unsqueeze(0), y=y1_example[0,...].unsqueeze(0)) # note here we use the last parameter estimate generated from a growing dataset
# adapted_weights, _ = model.inner_update_step_from_params(x=y0_example[0,...].unsqueeze(0), y=y1_example[0,...].unsqueeze(0), params=params) # note here we use the last parameter estimate generated from a growing dataset
# do the same with the online_adapt_maml call to make sure it works
data_stream = [(x.unsqueeze(0).unsqueeze(0),y.unsqueeze(0).unsqueeze(0)) for x, y in zip(y0_example[0,...], y1_example[0,...])]  # create a data stream from the example
adapted_weights2, _,_ = online_adapt_maml(model=model.model, loss_fn=model.loss_function, data_stream=data_stream, lr=model.internal_learning_rate, use_full_history=False)

# Roll out the model on the batch of initial conditions
# and compare to the ground truth trajectory
s = 0.1 # simulation time step
n = int(10 / s)  # number of steps to simulate

ground_truth_traj = [y0.clone()]
maml_traj = [y0.clone()]
maml_traj2 = [y0.clone()]  # for the second set of adapted weights
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

    maml_state2 = maml_traj2[-1]
    maml_dstate2 = model.forward(x=maml_state2, w=adapted_weights2)
    maml_prediction2 = maml_state2 + maml_dstate2
    maml_traj2.append(maml_prediction2) 


ground_truth_traj = torch.stack(ground_truth_traj, dim=0).detach().cpu().numpy()
maml_traj = torch.stack(maml_traj, dim=0).detach().cpu().numpy()
maml_traj2 = torch.stack(maml_traj2, dim=0).detach().cpu().numpy()

# Plot first 9 trajectories from the dataset, but pay attention to the first mu, since we fine-tuned on that
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
        (_m2,) = ax[i, j].plot(maml_traj2[:, traj_idx, 0, 0], maml_traj2[:, traj_idx, 0, 1], color='green', linestyle='--', label='MAML (online adapt)')
fig.legend(
    handles=[_t, _m, _m2],
    loc="outside upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=3,
    frameon=False,
)

# plt.show()
# save the figure
fig.savefig(f"./test.png", bbox_inches='tight')


"""
Quantitative evaluation of the MAML model.
"""
mu = torch.empty(1, device=device).uniform_(*dataset.mu_range) # random initial mu parameter
plotting_mu = [mu.item()]  # for plotting purposes, we will keep track of the mu parameter
losses_maml = []  # to store the losses for each step
adapted_weights = copy_model_params(model.model, 1)  # copy the parameters for each task in the batch, this is a placeholder for the first step
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

        # Compute the parameter estimate from data, using the last parameter estimate generated from a growing dataset
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

        tqdm_bar.set_postfix(
            {
                "loss_maml": f"{loss_maml.item():.2e}",
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
plt.legend()
plt.tight_layout()
# plt.show()
fig.savefig(f"./test_losses.png", bbox_inches='tight')

# # save the losses
# losses_maml = torch.tensor(losses_maml, device=device)
# torch.save({
#     "losses_maml": losses_maml,
# }, f"./logs/VanDerPol_{alg}/losses.pth")