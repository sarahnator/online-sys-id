import torch
from torch.utils.data import DataLoader

from function_encoder.coefficients import recursive_least_squares_update
from data.load_data import load_all_scenes
from torch.utils.data import IterableDataset
from typing import List

from models.maml import copy_model_params
from util.online_adapt import online_adapt_maml

from phoenix_util import *

import tqdm

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# device = "cpu"  # for testing purposes, we use CPU
torch.manual_seed(0)


class kStepTestDataset(IterableDataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
    ):
        self.inputs = inputs
        self.targets = targets
        self.n_example_points = n_example_points

    def __iter__(self):
        while True:
            # Traj length
            traj_length = 300

            # Sample random points from the data without replacement
            indices = torch.randperm(self.inputs[0].shape[0])
            example_indices = indices[: self.n_example_points]

            # Choose a random starting point for the trajectory
            # ensure that the trajectory length does not exceed the data length
            buffer = 50
            init_pt = torch.randint(self.inputs[0].shape[0] - (traj_length + buffer), (1,)).item()

            _xs = self.inputs[0][:, 1:]
            _dt = self.targets[0][:, 0] - self.inputs[0][:, 0]
            _ys = self.targets[0][:, 1:] - _xs[:, :6]
            example_xs = _xs[example_indices]
            example_dt = _dt[example_indices]
            example_ys = _ys[example_indices]

            x = self.inputs[0][init_pt:init_pt+traj_length, 1:7]
            dt = _dt[init_pt:init_pt+traj_length]
            u = self.inputs[0][init_pt:init_pt+traj_length, 7:9]
            y = self.targets[0][init_pt:init_pt+traj_length, 1:] - x

            yield x, dt, u, y, example_xs, example_dt, example_ys



# Load dataset

scene_data = load_all_scenes()
training_scenes = [0, 2, 3, 4, 6, 7]
interpolation_scenes = [5]
extrapolation_scenes = [1]
all_scenes = interpolation_scenes + training_scenes + extrapolation_scenes

# Preload all inputs/targets
scene_inputs, scene_targets = {}, {}
for idx in all_scenes:
    key = f"scene{idx}"
    scene_inputs[idx], scene_targets[idx] = scene_data[key]

# Choose the evaluation scene
scene = 3 # Only one scene for now

# Create a dataset for testing prediction errors. 
batch_size = 200
scene_input, scene_target = scene_inputs[scene], scene_targets[scene]
dataset = kStepTestDataset([scene_input], [scene_target], n_example_points=1000)
dataloader = DataLoader(dataset,batch_size=batch_size)
dataloader_iter = iter(dataloader)

# Load the model.
n_basis = 8 
seed = 42
model_path = f"logs/Phoenix_function_encoder/seed={seed}/function_encoder_model.pth"
model = load_model("function_encoder", device, n_basis, model_path)
node_model_path = f"logs/Phoenix_neural_ode/seed={seed}/neural_ode_model.pth"
node_model = load_model("neural_ode", device, n_basis, node_model_path)
maml_k1_model_path = f"logs/Phoenix_maml2_node/seed={seed}/maml2_node_model.pth"
maml_k1_model = load_model("maml2_node", device, n_basis, maml_k1_model_path)
maml_k100_model_path = f"logs/Phoenix_maml_node/seed={seed}/maml_node_model.pth"
maml_k100_model = load_model("maml_node", device, n_basis, maml_k100_model_path)


# Evaluate models
node_model.eval()
model.eval()
# with torch.no_grad():

# Initialize the coefficients, matching an assumed batch size of 1
coefficients = torch.zeros(batch_size, n_basis, device=device)
P = torch.eye(n_basis, device=device).repeat(batch_size,1,1)#.unsqueeze(0)

baseline_err_med = []
baseline_err_10th = []
baseline_err_90th = []

rls_err_med = []
rls_err_10th = []
rls_err_90th = []

node_err_med = []
node_err_10th = []
node_err_90th = []

node_sgd_err_med = []
node_sgd_err_10th = []
node_sgd_err_90th = []

node_sg_windowed_err_med = []
node_sg_windowed_err_10th = []
node_sg_windowed_err_90th = []

maml_k1_err_med = []
maml_k1_err_10th = []
maml_k1_err_90th = []

maml_k100_err_med = []
maml_k100_err_10th = []
maml_k100_err_90th = []

parameter_estimate_norms = []
parameter_estimate_stds = []

# Fetch new observation
batch = next(dataloader_iter)
x, dt, u, y, _, _, _ = batch
x = x.to(device)
dt = dt.to(device)
u = u.to(device)
y = y.to(device)

# Compute baseline coefficients
coefficients_baseline, _ = model.compute_coefficients((torch.cat((x, u), dim=-1), dt), y)

# set the SGD parameters
cumulative_data = []
# set the update MAML parameters
lr = 1e-3
window = 100  # use a window of 100 steps for the windowed version of maml

_adapted_params = copy_model_params(node_model, batch_size)  # copy the parameters for each task in the batch
_adapted_windowed_params = _adapted_params.copy()
_adapted_params_maml_k1 = copy_model_params(maml_k1_model.model, batch_size)
_adapted_params_maml_k100 = copy_model_params(maml_k100_model.model, batch_size)

with tqdm.trange(dt.shape[1]) as tqdm_bar:
    for step in tqdm_bar:
        # slice relevant quantities at the current step
        dt_step = dt[:, step].unsqueeze(1)#.unsqueeze(0)
        y_step = y[:, step].unsqueeze(1)
        x_step = x[:, step].unsqueeze(1)  # x is constant for the trajectory
        u_step = u[:, step].unsqueeze(1)  # u is constant for the trajectory

        # Append the observation to the cumulative data
        cumulative_data.append(((torch.cat((x_step,u_step), dim=-1), dt_step), y_step))

        adapted_params, loss, updates = online_adapt_maml(
            model=node_model,
            loss_fn=node_model.loss_fn,
            data_stream=[((torch.cat((x_step,u_step), dim=-1), dt_step), y_step)],  # single observation
            lr=lr,
            use_full_history=False,  # we only use the single observation
            _params=_adapted_params,  # use the current adapted parameters
        )

        # Windowed version of maml
        windowed_data = cumulative_data[-window:] if len(cumulative_data) > window else cumulative_data
        windowed_adapted_params, windowed_loss, _ = online_adapt_maml(
            model=node_model,
            loss_fn=node_model.loss_fn,
            data_stream=windowed_data,  # last `window` observations
            lr=lr,
            use_full_history=True,  # we use all observations in the window
            _params=_adapted_windowed_params,  # use the current adapted windowed parameters
        )

        adapted_params_maml_k1, maml_k1_loss  = maml_k1_model.inner_update_step_from_params(
            x=(torch.cat((x_step,u_step), dim=-1)),
            dt=dt_step,
            y=y_step,
            params=_adapted_params_maml_k1,
        )

        start = max(0, step - window + 1)
        adapted_params_maml_k100, maml_k100_loss = maml_k100_model.inner_update_step_from_params(
            x=(torch.cat((x[:, start:step+1], u[:, start:step+1]), dim=-1)),
            dt=dt[:, start:step+1],
            y=y[:, start:step+1],
            params=_adapted_params_maml_k100,
        )

        # Compute the basis functions 
        # [batch_size, n_points, n_features, n_basis]
        g = model.basis_functions((torch.cat((x_step,u_step), dim=-1), dt_step))

        L = torch.linalg.cholesky(P)
        coefficients, P = recursive_least_squares_update(
            method='qr',
            g=g,
            y=y_step,
            P=L,
            coefficients=coefficients,
            forgetting_factor=0.95,
        )


        # Compute the baseline error
        pred_baseline = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients_baseline)
        loss_baseline = torch.nn.functional.mse_loss(pred_baseline, y_step)

        # Compute the recursive least squares prediction error
        pred = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients)
        loss_rls = torch.nn.functional.mse_loss(pred, y_step)

        # Compute the neural ODE error
        node_pred = node_model((torch.cat((x_step, u_step), dim=-1), dt_step))
        loss_node = torch.nn.functional.mse_loss(node_pred, y_step)

        # Compute the neural ODE prediction error with SGD
        node_sgd_pred = node_model.forward((torch.cat((x_step,u_step), dim=-1), dt_step), {'params': adapted_params})
        loss_node_sgd = torch.nn.functional.mse_loss(node_sgd_pred, y_step)

        # Compute node predictions with windowed update
        windowed_node_sgd_pred = node_model.forward((torch.cat((x_step,u_step), dim=-1), dt_step), {'params': windowed_adapted_params})
        windowed_sgd_loss = torch.nn.functional.mse_loss(windowed_node_sgd_pred, y_step)

        # Compute MAML predictions
        maml_k1_pred = maml_k1_model.forward((torch.cat((x_step, u_step), dim=-1), dt_step), model_kwargs={'params': adapted_params_maml_k1})
        maml_k1_loss = torch.nn.functional.mse_loss(maml_k1_pred, y_step)
        maml_k100_pred = maml_k100_model.forward((torch.cat((x_step, u_step), dim=-1), dt_step), model_kwargs={'params': adapted_params_maml_k100})
        maml_k100_loss = torch.nn.functional.mse_loss(maml_k100_pred, y_step)

        baseline_err_med.append(torch.norm(y_step - pred_baseline, dim=-1).median().item())
        baseline_err_10th.append(torch.norm(y_step - pred_baseline, dim=-1).quantile(0.10).item())
        baseline_err_90th.append(torch.norm(y_step - pred_baseline, dim=-1).quantile(0.90).item())

        rls_err_med.append(torch.norm(y_step - pred, dim=-1).median().item())
        rls_err_10th.append(torch.norm(y_step - pred, dim=-1).quantile(0.10).item())
        rls_err_90th.append(torch.norm(y_step - pred, dim=-1).quantile(0.90).item())

        node_err_med.append(torch.norm(y_step - node_pred, dim=-1).median().item())
        node_err_10th.append(torch.norm(y_step - node_pred, dim=-1).quantile(0.10).item())
        node_err_90th.append(torch.norm(y_step - node_pred, dim=-1).quantile(0.90).item())

        parameter_estimate_norms.append((coefficients - coefficients_baseline).norm(dim=-1).mean().item())
        parameter_estimate_stds.append((coefficients - coefficients_baseline).norm(dim=-1).std().item())

        node_sgd_err_med.append(torch.norm(y_step - node_sgd_pred, dim=-1).median().item())
        node_sgd_err_10th.append(torch.norm(y_step - node_sgd_pred, dim=-1).quantile(0.10).item())
        node_sgd_err_90th.append(torch.norm(y_step - node_sgd_pred, dim=-1).quantile(0.90).item())  

        node_sg_windowed_err_med.append(torch.norm(y_step - windowed_node_sgd_pred, dim=-1).median().item())
        node_sg_windowed_err_10th.append(torch.norm(y_step - windowed_node_sgd_pred, dim=-1).quantile(0.10).item())
        node_sg_windowed_err_90th.append(torch.norm(y_step - windowed_node_sgd_pred, dim=-1).quantile(0.90).item())

        maml_k1_err_med.append(torch.norm(y_step - maml_k1_pred, dim=-1).median().item())
        maml_k1_err_10th.append(torch.norm(y_step - maml_k1_pred, dim=-1).quantile(0.10).item())
        maml_k1_err_90th.append(torch.norm(y_step - maml_k1_pred, dim=-1).quantile(0.90).item())

        maml_k100_err_med.append(torch.norm(y_step - maml_k100_pred, dim=-1).median().item())
        maml_k100_err_10th.append(torch.norm(y_step - maml_k100_pred, dim=-1).quantile(0.10).item())
        maml_k100_err_90th.append(torch.norm(y_step - maml_k100_pred, dim=-1).quantile(0.90).item())

        # set the adapted parameters for the next step
        _adapted_params = adapted_params.copy()
        _adapted_windowed_params = windowed_adapted_params.copy()

        _adapted_params_maml_k1 = adapted_params_maml_k1.copy()
        _adapted_params_maml_k100 = adapted_params_maml_k100.copy()


# Use STIX fonts (LaTeX-style) and apply them consistently
plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Plot the losses
fig1 = plt.figure(figsize=(3.5, 2.5))

plt.plot(node_err_med, label="NODE", color="#D62728")
plt.fill_between(
    range(len(node_err_med)),
    node_err_10th,
    node_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#D62728"
)

plt.plot(baseline_err_med, label="FE", color="#1F77B4")
plt.fill_between(
    range(len(baseline_err_med)),
    baseline_err_10th,
    baseline_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#1F77B4"
)

plt.plot(rls_err_med, label="FE-RLS", color="#2ca02c")
plt.fill_between(
    range(len(rls_err_med)),
    rls_err_10th,
    rls_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#2ca02c"
)

plt.plot(node_sgd_err_med, label="NODE-SGD", color="#9467bd", linestyle='--')
plt.fill_between(
    range(len(node_sgd_err_med)),
    node_sgd_err_10th,
    node_sgd_err_90th,
    alpha=0.2,
    edgecolor='none',   
    linewidth=0.0,
    color="#9467bd"
)

plt.plot(node_sg_windowed_err_med, label="NODE-SGD-W", color="#8c564b", linestyle=':')
plt.fill_between(
    range(len(node_sg_windowed_err_med)),
    node_sg_windowed_err_10th,      
    node_sg_windowed_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,  
    color="#8c564b"
)

plt.plot(maml_k1_err_med, label="MAML-1", color="#e377c2", linestyle='-.')
plt.fill_between(
    range(len(maml_k1_err_med)),
    maml_k1_err_10th,
    maml_k1_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#e377c2"
)

plt.plot(maml_k100_err_med, label="MAML-100", color="#7f7f7f", linestyle=':')
plt.fill_between(
    range(len(maml_k100_err_med)),
    maml_k100_err_10th,
    maml_k100_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#7f7f7f"
)

plt.yscale("log")
plt.xlabel("Time Steps")
plt.ylabel("Single Step Mean Prediction Error")

fig1.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    frameon=False,
)

plt.tight_layout()
plt.savefig(f"plots/rls_one_terrain/scene={scene}", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()


fig2, ax2 = plt.subplots(figsize=(3.5,2.5))

ax2.plot(parameter_estimate_norms)
ax2.fill_between(
    range(len(parameter_estimate_stds)),
    [x - y for x, y in zip(parameter_estimate_norms, parameter_estimate_stds)],
    [x + y for x, y in zip(parameter_estimate_norms, parameter_estimate_stds)],
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
)
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Normed Coeffecient Error")
fig2.tight_layout()
plt.savefig(f"plots/rls_one_terrain/scene={scene}_coeffs", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()