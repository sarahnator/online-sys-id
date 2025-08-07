import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_scenes, MultiRolloutDataset
from models import maml
from train_utils import inertial_to_body, inertial_to_body_XL
from torch.utils.data import DataLoader
from function_encoder.coefficients import recursive_least_squares_update

from models.maml import copy_model_params
from util.online_adapt import online_adapt_maml

from phoenix_util import *

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Config
n_basis = 8
k_steps = 30
n_rollouts = 100
batchsize = 100
torch.manual_seed(30)
# seeds = list(range(10))
seeds = [42]
model_types = ["function_encoder", "neural_ode"]  

# Choose the evaluation scene
scene = 1
scene_data = load_scenes([scene])

# Create a dataset for testing prediction errors. 
scene_input, scene_target = scene_data[f'scene{scene}']
dataset = MultiRolloutDataset(
    inputs=[scene_input],
    targets=[scene_target],
    n_example_points=100,
    k_steps=k_steps,
    n_rollouts=n_rollouts,
)
dataloader = DataLoader(dataset, batch_size=batchsize)

# Evaluate
all_results = {mt: {seed: [] for seed in seeds} for mt in model_types}
rls_results = {seed: [] for seed in seeds}
node_sgd_results = {seed: [] for seed in seeds}
windowed_node_sgd_results = {seed: [] for seed in seeds}
maml_k1_results = {seed: [] for seed in seeds}
maml_k100_results = {seed: [] for seed in seeds}

# with torch.no_grad():
for seed in seeds:
    
    x0_seq, dt_seq, u_seq, y_seq, ex_xs, ex_dt, ex_ys = next(iter(dataloader))
    x0_seq, dt_seq, u_seq, y_seq = [t.to(device) for t in [x0_seq, dt_seq, u_seq, y_seq]]
    ex_xs, ex_dt, ex_ys = [t.to(device) for t in [ex_xs, ex_dt, ex_ys]]

    with torch.no_grad():
        for mt in model_types:
            # Load the model. 
            model_path = f"logs/Phoenix_{mt}/seed={seed}/{mt}_model.pth"
            if not os.path.exists(model_path):
                print(f"Missing: {model_path}")
                exit()
            # model, loss_fn = load_model(mt, device, n_basis, model_path)
            model = load_model(mt, device, n_basis, model_path)

            if mt == "function_encoder":
                # Compute the basis coefficients
                coefficients, _ = model.compute_coefficients((ex_xs, ex_dt), ex_ys)
            
            # Make a copy of the current state for processing.
            _x = x0_seq.clone()
            total_error = torch.zeros((batchsize, n_rollouts), device=device)

            # Loop over the k steps and predict the next state.
            for k in range(k_steps):

                # Predict the next state and save the prediction. 
                if mt == "function_encoder":
                    del_x = model((torch.cat((_x, u_seq[:,:,k,:]), dim=2), dt_seq[:,:,k]), coefficients=coefficients)
                elif mt == "neural_ode":
                    del_x = model((torch.cat((_x, u_seq[:,:,k,:]), dim=2), dt_seq[:,:,k]))

                # Get the next velocity in the initial body frame.
                next_vel_Bi = _x[:,:,3:6] + del_x[:,:,3:6]

                # Transform the velocity back to the body frame.
                next_vel_B = inertial_to_body_XL(
                    bIMat=del_x[:,:,:3],
                    xIMat=next_vel_Bi,
                    device=device
                )

                # Prepare the new current state. 
                _x = torch.cat((torch.zeros((batchsize, n_rollouts, 3), device=device), next_vel_B), dim=-1)
                
                # Calculate and accumulate the error. 
                pred = torch.cat((del_x[:,:,:3], next_vel_Bi), dim=-1)
                # total_error += torch.nn.functional.mse_loss(pred, y_seq[:,:,k,:])
                total_error += torch.norm(y_seq[:,:,k,:] - pred, dim=-1)

            # Save results from this model. 
            all_results[mt][seed] = total_error.cpu().numpy()

    # Initialize the RLS problem.
    with torch.no_grad():
        model_path = f"logs/Phoenix_function_encoder/seed={seed}/function_encoder_model.pth"
        rls_model = load_model("function_encoder", device, n_basis, model_path)
        P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
        coeffs = torch.zeros(batchsize, n_basis, device=device)
        rls_error = torch.zeros((batchsize, n_rollouts), device=device)

        for i in range(n_rollouts):
            # Get the next point in the RLS update data.
            x_step = x0_seq[:, i, :].unsqueeze(1)
            u_step = u_seq[:, i, 0, :].unsqueeze(1)
            dt_step = dt_seq[:, i, 0].unsqueeze(1)
            y_step = y_seq[:, i, 0, :].unsqueeze(1) - x_step

            # Compute new coeffs using RLS update. 
            g = rls_model.basis_functions((torch.cat((x_step, u_step), dim=-1), dt_step))
            L = torch.linalg.cholesky(P)
            coeffs, P = recursive_least_squares_update(
                method='qr', g=g, y=y_step, P=L, coefficients=coeffs, forgetting_factor=0.95
            )

            _x = x0_seq[:,i,:].clone()
            for k in range(k_steps):
                
                print(f"seed={seed}, rollout={i}, step={k}")

                # Predict the next state and save the prediction. 
                del_x = rls_model((torch.cat((_x.unsqueeze(1), u_seq[:,i,k,:].unsqueeze(1)), dim=-1),
                                    dt_seq[:,i,k].unsqueeze(1)), coefficients=coeffs)

                # Get the next velocity in the initial body frame.
                next_vel_Bi = _x[:,3:6] + del_x[:,:,3:6].squeeze(1)

                # Transform the velocity back to the body frame.
                next_vel_B = inertial_to_body(
                    bIMat=del_x[:,:,:3].squeeze(1),
                    xIMat=next_vel_Bi,
                    device=device
                )

                # Prepare the new current state. 
                _x = torch.cat((torch.zeros((batchsize, 3), device=device), next_vel_B), dim=-1)

                # Calculate and accumulate the error. 
                pred = torch.cat((del_x[:,:,:3].squeeze(1), next_vel_Bi), dim=-1)
                # rls_error[:,i] += torch.nn.functional.mse_loss(pred, y_seq[:,i,k,:])
                rls_error[:,i] += torch.norm(y_seq[:,i,k,:] - pred, dim=-1)

            # Save the results from this rollout.
            rls_results[seed] = rls_error.cpu().numpy()

    # Initialize the RLS problem.
    # model_path = f"logs/Phoenix_function_encoder/seed={seed}/function_encoder_model.pth"
    # rls_model = load_model("function_encoder", device, n_basis, model_path)
    # P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
    # coeffs = torch.zeros(batchsize, n_basis, device=device)
    # rls_error = torch.zeros((batchsize, n_rollouts), device=device)
    node_sgd_error = torch.zeros((batchsize, n_rollouts), device=device)
    windowed_node_sgd_error = torch.zeros((batchsize, n_rollouts), device=device)
    maml_k1_error = torch.zeros((batchsize, n_rollouts), device=device)
    maml_k100_error = torch.zeros((batchsize, n_rollouts), device=device)

    # SGD-based algorithms
    # set the SGD parameters
    cumulative_data = []
    # set the update MAML parameters
    lr = 10
    window = 100  # use a window of 100 steps for the windowed version of maml
    node_model = load_model("neural_ode", device, n_basis, f"logs/Phoenix_neural_ode/seed={seed}/neural_ode_model.pth")
    maml_k1_model = load_model("maml_node", device, n_basis, f"logs/Phoenix_maml_node/seed={seed}/maml_node_model.pth")
    maml_k100_model = load_model("maml2_node", device, n_basis, f"logs/Phoenix_maml2_node/seed={seed}/maml2_node_model.pth")
    
    _adapted_params = copy_model_params(node_model, batchsize)  # copy the parameters for each task in the batch
    _adapted_windowed_params = _adapted_params.copy()
    _adapted_params_maml_k1 = copy_model_params(maml_k1_model.model, batchsize)
    _adapted_params_maml_k100 = copy_model_params(maml_k100_model.model, batchsize)

    for i in range(n_rollouts):

        # Get the next point in the RLS update data.
        x_step = x0_seq[:, i, :].unsqueeze(1)
        u_step = u_seq[:, i, 0, :].unsqueeze(1)
        dt_step = dt_seq[:, i, 0].unsqueeze(1)
        y_step = y_seq[:, i, 0, :].unsqueeze(1) - x_step

        # # Compute new coeffs using RLS update. 
        # g = rls_model.basis_functions((torch.cat((x_step, u_step), dim=-1), dt_step))
        # L = torch.linalg.cholesky(P)
        # coeffs, P = recursive_least_squares_update(
        #     method='qr', g=g, y=y_step, P=L, coefficients=coeffs, forgetting_factor=0.95
        # )

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
        adapted_windowed_params, windowed_loss, _ = online_adapt_maml(
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

        start = max(0, i - window + 1)
        adapted_params_maml_k100, maml_k100_loss = maml_k100_model.inner_update_step_from_params(
            x=(torch.cat((x0_seq[:, start:i+1, :], u_seq[:, start:i+1, 0, :]), dim=-1)),
            dt=dt_seq[:, start:i+1, 0],
            y=y_seq[:, start:i+1, 0, :] - x0_seq[:, start:i+1, :],
            params=_adapted_params_maml_k100,
        )

        # Make predictions with the adapted parameters

        # _x = x0_seq[:,i,:].clone()
        _x_node_sgd = x0_seq[:,i,:].clone()
        _x_windowed_node_sgd = x0_seq[:,i,:].clone()
        _x_maml_k1 = x0_seq[:,i,:].clone()
        _x_maml_k100 = x0_seq[:,i,:].clone()
        for k in range(k_steps):
            
            print(f"seed={seed}, rollout={i}, step={k}")

            # # Predict the next state and save the prediction. 
            # del_x = rls_model((torch.cat((_x.unsqueeze(1), u_seq[:,i,k,:].unsqueeze(1)), dim=-1),
            #                     dt_seq[:,i,k].unsqueeze(1)), coefficients=coeffs)

            # Use the adapted parameters to predict the next state.
            # Compute the neural ODE prediction error with SGD
            node_sgd_del_x = node_model.forward((torch.cat((_x_node_sgd.unsqueeze(1), u_seq[:,i,k,:].unsqueeze(1)), dim=-1), dt_seq[:,i,k].unsqueeze(1)), {'params': adapted_params})

            # Compute node predictions with windowed update
            windowed_node_sgd_del_x = node_model.forward((torch.cat((_x_windowed_node_sgd.unsqueeze(1), u_seq[:,i,k,:].unsqueeze(1)), dim=-1), dt_seq[:,i,k].unsqueeze(1)), {'params': adapted_windowed_params})

            # Compute MAML predictions
            maml_k1_del_x = maml_k1_model.forward((torch.cat((_x_maml_k1.unsqueeze(1), u_seq[:,i,k,:].unsqueeze(1)), dim=-1), dt_seq[:,i,k].unsqueeze(1)), model_kwargs={'params': adapted_params_maml_k1})
            maml_k100_del_x = maml_k100_model.forward((torch.cat((_x_maml_k100.unsqueeze(1), u_seq[:,i,k,:].unsqueeze(1)), dim=-1), dt_seq[:,i,k].unsqueeze(1)), model_kwargs={'params': adapted_params_maml_k100})

            # Get the next velocity in the initial body frame.
            # next_vel_Bi = _x[:,3:6] + del_x[:,:,3:6].squeeze(1)

            next_vel_Bi_node_sgd = _x_node_sgd[:,3:6] + node_sgd_del_x[:,:,3:6].squeeze(1)
            next_vel_Bi_windowed_node_sgd = _x_windowed_node_sgd[:,3:6] + windowed_node_sgd_del_x[:,:,3:6].squeeze(1)
            next_vel_Bi_maml_k1 = _x_maml_k1[:,3:6] + maml_k1_del_x[:,:,3:6].squeeze(1)
            next_vel_Bi_maml_k100 = _x_maml_k100[:,3:6] + maml_k100_del_x[:,:,3:6].squeeze(1)

            # Transform the velocity back to the body frame.
            # next_vel_B = inertial_to_body(
            #     bIMat=del_x[:,:,:3].squeeze(1),
            #     xIMat=next_vel_Bi,
            #     device=device
            # )

            next_vel_B_node_sgd = inertial_to_body(
                bIMat=node_sgd_del_x[:,:,:3].squeeze(1),
                xIMat=next_vel_Bi_node_sgd,
                device=device
            )   
            next_vel_B_windowed_node_sgd = inertial_to_body(
                bIMat=windowed_node_sgd_del_x[:,:,:3].squeeze(1),
                xIMat=next_vel_Bi_windowed_node_sgd,
                device=device
            )

            next_vel_B_maml_k1 = inertial_to_body(
                bIMat=maml_k1_del_x[:,:,:3].squeeze(1),
                xIMat=next_vel_Bi_maml_k1,
                device=device
            )
            next_vel_B_maml_k100 = inertial_to_body(
                bIMat=maml_k100_del_x[:,:,:3].squeeze(1),
                xIMat=next_vel_Bi_maml_k100,
                device=device
            )

            # Prepare the new current state. 
            # _x = torch.cat((torch.zeros((batchsize, 3), device=device), next_vel_B), dim=-1)

            _x_node_sgd = torch.cat((torch.zeros((batchsize, 3), device=device), next_vel_B_node_sgd), dim=-1)  
            _x_windowed_node_sgd = torch.cat((torch.zeros((batchsize, 3), device=device), next_vel_B_windowed_node_sgd), dim=-1)
            _x_maml_k1 = torch.cat((torch.zeros((batchsize, 3), device=device), next_vel_B_maml_k1), dim=-1)
            _x_maml_k100 = torch.cat((torch.zeros((batchsize, 3), device=device), next_vel_B_maml_k100), dim=-1)    

            # Calculate and accumulate the error. 
            # pred = torch.cat((del_x[:,:,:3].squeeze(1), next_vel_Bi), dim=-1)

            pred_node_sgd = torch.cat((node_sgd_del_x[:,:,:3].squeeze(1), next_vel_Bi_node_sgd), dim=-1)
            pred_windowed_node_sgd = torch.cat((windowed_node_sgd_del_x[:,:,:3].squeeze(1), next_vel_Bi_windowed_node_sgd), dim=-1)
            pred_maml_k1 = torch.cat((maml_k1_del_x[:,:,:3].squeeze(1), next_vel_Bi_maml_k1), dim=-1)
            pred_maml_k100 = torch.cat((maml_k100_del_x[:,:,:3].squeeze(1), next_vel_Bi_maml_k100), dim=-1)
            
            # Accumulate the errors for the RLS method
            # rls_error[:,i] += torch.norm(y_seq[:,i,k,:] - pred, dim=-1)

            # Accumulate the errors for the SGD-based methods
            node_sgd_error[:,i] += torch.norm(y_seq[:,i,k,:] - pred_node_sgd, dim=-1)
            windowed_node_sgd_error[:,i] += torch.norm(y_seq[:,i,k,:] - pred_windowed_node_sgd, dim=-1)
            maml_k1_error[:,i] += torch.norm(y_seq[:,i,k,:] - pred_maml_k1, dim=-1)
            maml_k100_error[:,i] += torch.norm(y_seq[:,i,k,:] - pred_maml_k100, dim=-1)

        # continually update the adapted parameters
        _adapted_params = adapted_params
        _adapted_windowed_params = adapted_windowed_params  
        _adapted_params_maml_k1 = adapted_params_maml_k1
        _adapted_params_maml_k100 = adapted_params_maml_k100

        # Save the results from this rollout.
        node_sgd_results[seed] = node_sgd_error.cpu().detach().numpy()
        windowed_node_sgd_results[seed] = windowed_node_sgd_error.cpu().detach().numpy()
        maml_k1_results[seed] = maml_k1_error.cpu().detach().numpy()
        maml_k100_results[seed] = maml_k100_error.cpu().detach().numpy()
        # rls_results[seed] = rls_error.cpu().detach().numpy()



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

# Plotting
fig = plt.figure(figsize=(3.5, 2.5))
colors = {"neural_ode": "#D62728", "function_encoder": "#1F77B4", "rls": "#2ca02c", "node_sgd": "#9467bd", "windowed_node_sgd": "#8c564b", "maml_k1": "#e377c2", "maml_k100": "#7f7f7f"}
names = {"neural_ode": "Neural ODE", "function_encoder": "Function Encoder", "rls": "FE-RLS", "node_sgd": "Node SGD", "windowed_node_sgd": "Windowed Node SGD", "maml_k1": "MAML-K1", "maml_k100": "MAML-K100"}
linestyles = {"neural_ode": "-", "function_encoder": "--", "rls": "-.", "node_sgd": "-", "windowed_node_sgd": "--", "maml_k1": ":", "maml_k100": "-."}
for mt in model_types + ["rls", "node_sgd", "windowed_node_sgd", "maml_k1", "maml_k100"]:
# for mt in ["node_sgd", "windowed_node_sgd", "maml_k1", "maml_k100"]:
    # Collect the final accumulated errors from all seeds and all rollouts
    if mt == "rls":
        errors = np.concatenate([rls_results[seed] for seed in seeds], axis=0)
    elif mt == "node_sgd":
        errors = np.concatenate([node_sgd_results[seed] for seed in seeds], axis=0) 
    elif mt == "windowed_node_sgd":
        errors = np.concatenate([windowed_node_sgd_results[seed] for seed in seeds], axis=0)
    elif mt == "maml_k1":
        errors = np.concatenate([maml_k1_results[seed] for seed in seeds], axis=0)
    elif mt == "maml_k100":
        errors = np.concatenate([maml_k100_results[seed] for seed in seeds], axis=0)
    else:
        errors = np.concatenate([all_results[mt][seed] for seed in seeds], axis=0)

    # Compute statistics
    med = np.median(errors, axis=0)
    _min = np.percentile(errors, 10, axis=0)
    _max = np.percentile(errors, 90, axis=0)

    # Plot median, min, and max. 
    plt.plot(med, label=names[mt], color=colors[mt], linestyle=linestyles[mt])
    plt.fill_between(
        np.arange(n_rollouts),
        _min,
        _max,
        alpha=0.2,
        color=colors[mt],
        edgecolor="none",
        linewidth=0.0,
    )

plt.yscale("log")
plt.xlabel("Number of Time Steps")
plt.ylabel(f"Accumulated Rollout Error")
fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    frameon=False,
)
plt.tight_layout()
plt.savefig(f"rls_error_k={k_steps}_step_scene_{scene}_log.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()