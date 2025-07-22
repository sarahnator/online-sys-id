from typing import Callable, Optional, Tuple, Union

from sympy import expand
from util.losses import neural_ode_loss
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
alg = 'MAML2_NODE'
model = get_model(algorithm=alg, device=device)
model.load_state_dict(torch.load(f"./logs/VanDerPol_{alg}/model.pth", map_location=device))
model.loss_function = torch.nn.MSELoss()
model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


"""
Qualitative evaluation of the MAML model.
"""

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
fig.savefig(f"./logs/VanDerPol_{alg}/qualitative_example2.png", bbox_inches='tight')
plt.close(fig)

"""
Quantitative evaluation of the MAML model.
"""
losses_maml = []  # to store the losses for each step

# roll out ground truth trajectories where mu changes every 200 steps
n_trials = 100

s = 0.1 # simulation time step
traj_len = int(150 / s)  # number of steps to simulate
change_mu_every = 500
traj_mus = torch.empty(traj_len // change_mu_every, device=device).uniform_(*dataset.mu_range)
trajectories = [torch.empty(n_trials, 1, 2, device=device).uniform_(*dataset.y0_range).clone()]
dt = torch.tensor([s], device=device).expand(trajectories[0].shape[0], -1).expand(-1, trajectories[0].shape[1])  # time step for simulation
# compute ground truth trajectories
for i in range(traj_len):
    state = trajectories[-1]
    mu = traj_mus[i // change_mu_every] if i % change_mu_every == 0 else mu
    change_in_state = rk4_step(van_der_pol, state, dt, mu=mu)
    resulting_state = state + change_in_state
    trajectories.append(resulting_state)
trajectories = torch.stack(trajectories, dim=1) # shape (n_trials, traj_len, 2)

## visualize the time-varying trajectories
# fig, ax = plt.subplots(3, 3, figsize=(10, 10))
# for i in range(3):
#     for j in range(3):
#         traj_idx = i * 3 + j
#         _mu = traj_mus[i * 3 + j]
#         ax[i, j].set_title(f"$\\mu$={_mu.item():.1f}")
#         ax[i, j].set_xlim(-5, 5)
#         ax[i, j].set_ylim(-5, 5)
#         (_t,) = ax[i, j].plot(trajectories[traj_idx, :, 0, 0].cpu(), trajectories[traj_idx, :, 0, 1].cpu(), color='blue', label='True')

losses = []

# make a prediction with the base model parameters
y0 = trajectories[:, 0, :]  # initial condition for the trials
y1_true = trajectories[:, 1, :]  # ground truth next state
init_params = copy_model_params(model.model, n_trials)  # initial parameters for the model
pred = y0 + model.forward((y0, dt), {"params": init_params})  # make a prediction with the base model parameters
losses.append(model.loss_function(pred, y1_true).item())
one_step_predicted_states = [y0.detach().cpu() + pred.detach().cpu()]
predicted_trajectories = one_step_predicted_states.copy()  # store the initial prediction
all_parameters = [init_params]
 
for i in range(1, traj_len):
    # Get the mu parameter for this step
    mu = traj_mus[i // change_mu_every] if i % change_mu_every == 0 else mu

    # Get ground truth observation from the trajectory
    y0 = trajectories[:, i - 1, :]
    y1 = trajectories[:, i, :]  # ground truth next state

    # update the model with the new observation
    adapted_weights, _ = model.inner_update_step(x=y0, dt=dt, y=y1)
    all_parameters.append(adapted_weights)  # store the adapted weights
    # Compute maml predictions with update on the single observation
    maml_pred = y0 + model.forward((y0, dt), model_kwargs={'params': adapted_weights})
    one_step_predicted_states.append(maml_pred.detach().cpu())
    loss = model.loss_function(maml_pred, y1)
    losses.append(loss.item())

    # roll out the predicted trajectory from previous maml prediction
    prev = predicted_trajectories[-1].to(device)  # previous prediction
    next_pred = prev + model.forward((prev, dt), model_kwargs={'params': adapted_weights})
    predicted_trajectories.append(next_pred.detach().cpu())

one_step_predicted_states = torch.stack(one_step_predicted_states, axis=1)  # shape (n_trials, traj_len, 2)
predicted_trajectories = torch.stack(predicted_trajectories, axis=1)  # shape (n_trials, traj_len, 2)

# Plot the loss
fig, ax = plt.subplots(1, 1, figsize=(10,10))

for i, m in enumerate(list(traj_mus)):
    x = i * change_mu_every
    ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    ax.text(
        x,               # data x
        0.1,               # axis-fraction y = 0 (bottom of the plotting area)
        f"$\\mu$={m:.1f}",
        transform=ax.get_xaxis_transform(),  # <-- key!
        rotation=90,
        va='bottom',     # push the text upward from the axis spine
        ha='left',
        fontsize=11,
        fontweight='bold',
    )
    
ax.set_yscale("log")
# ax.set_yscale("log")
ax.minorticks_on()
ax.grid(which="both", axis="y", linestyle=":", linewidth=0.5)
ax.plot(losses, label="MAML NODE Loss", color='blue')
plt.legend()
plt.tight_layout()
# plt.show()
fig.savefig(f"./logs/VanDerPol_{alg}/losses_example2.png", bbox_inches='tight')

# save the losses
losses_maml = torch.tensor(losses_maml, device=device)
torch.save({
    "losses_maml": losses_maml,
}, f"./logs/VanDerPol_{alg}/losses_example2.pth")

# visualize the predicted trajectories
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        traj_idx = i * 3 + j
        # _mu = traj_mus[i * 3 + j]
        # ax[i, j].set_title(f"$\\mu$={_mu.item():.1f}")
        ax[i, j].set_xlim(-5, 5)
        ax[i, j].set_ylim(-5, 5)
        (_t,) = ax[i, j].plot(trajectories[traj_idx, :, 0, 0].cpu(), trajectories[traj_idx, :, 0, 1].cpu(), color='blue', label='True')
        (_m,) = ax[i, j].plot(
            one_step_predicted_states[traj_idx, :, 0, 0].cpu(),
            one_step_predicted_states[traj_idx, :, 0, 1].cpu(),
            color='orange',
            label='MAML Single Step Prediction'
        )
        (_p, ) = ax[i, j].plot(
            predicted_trajectories[traj_idx, :, 0, 0].cpu(),
            predicted_trajectories[traj_idx, :, 0, 1].cpu(),
            color='green',
            label='MAML Prediction'
        )
fig.legend(
    handles=[_t, _m, _p],
    loc="outside upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=3,
    frameon=False,
)

# save the figure
fig.savefig(f"./logs/VanDerPol_{alg}/trajectories_example2.png", bbox_inches='tight')

"""
How do the trajectory predictions look like as we continue to adapt the model parameters?
"""
indices = torch.arange(0, traj_len, change_mu_every, device=device)  # indices where mu changes

_adapted_weights = [all_parameters[i]for i in indices]  # take the parameters from the first trial where mu changes
# grab only the first trial

# sample an initial condition for the trajectory
_y0 = torch.empty(1, 1, 2, device=device).uniform_(*dataset.y0_range).clone()
_true_traj_per_mu = [torch.empty(traj_len, 2, device=device) for _ in range(traj_mus.shape[0])]
_predicted_traj_per_mu = [torch.empty(traj_len, 2, device=device) for _ in range(traj_mus.shape[0])]
for i, mu in enumerate(traj_mus):
    _y0[:, 0, 0] = _y0[:, 0, 1] = mu.item()  # set the initial condition to the current mu value
    _dt = torch.tensor([s], device=device).expand(_y0.shape[0], -1).expand(-1, _y0.shape[1])  # time step for simulation
    _true_traj_per_mu[i][0] = _y0  # set the first state to the initial condition
    _predicted_traj_per_mu[i][0] = _y0  # set the first state to the initial condition

    for j in range(1, traj_len):
        change_in_state = rk4_step(van_der_pol, _true_traj_per_mu[i][j - 1], _dt, mu=mu)
        resulting_state = _true_traj_per_mu[i][j - 1] + change_in_state
        _true_traj_per_mu[i][j] = resulting_state
        maml_dstate = model.forward(inputs=(_predicted_traj_per_mu[i][j - 1].unsqueeze(0).expand(n_trials, -1, -1), _dt.expand(n_trials, -1)), model_kwargs={'params': _adapted_weights[i]})[0,:,:]  # get the maml prediction for the current mu for just the first trial
        maml_prediction = _predicted_traj_per_mu[i][j - 1] + maml_dstate
        _predicted_traj_per_mu[i][j] = maml_prediction
_true_traj_per_mu = torch.stack(_true_traj_per_mu, dim=0)  # shape (n_mus, traj_len, 2)
_predicted_traj_per_mu = torch.stack(_predicted_traj_per_mu, dim=0)  # shape (n_mus, traj_len, 2)
# Plot the predicted trajectories for each mu
fig, ax = plt.subplots(traj_mus.shape[0], traj_mus.shape[0], figsize=(10, 10))
for _mu in range(traj_mus.shape[0]): # mu
    for _parameter_update in range(traj_mus.shape[0]):  # parameter update for each mu
        ax[_mu, _parameter_update].set_xlim(-5, 5)
        ax[_mu, _parameter_update].set_ylim(-5, 5)
        (_t,) = ax[_mu, _parameter_update].plot(_true_traj_per_mu[_mu, :, 0].detach().cpu(), _true_traj_per_mu[_mu, :, 1].detach().cpu(), color='blue', label='True')
        (_m,) = ax[_mu, _parameter_update].plot(
            _predicted_traj_per_mu[_parameter_update, :, 0].detach().cpu(),
            _predicted_traj_per_mu[_parameter_update, :, 1].detach().cpu(),
            color='orange',
            label='MAML Prediction'
        )
fig.legend(
    handles=[_t, _m],
    loc="outside upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    frameon=False,
)

# save the figure
fig.savefig(f"./logs/VanDerPol_{alg}/all_systems_prediction_example2.png", bbox_inches='tight')

print(traj_mus)