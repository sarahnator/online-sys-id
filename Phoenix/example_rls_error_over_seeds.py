import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.load_data import TestDataset
from train_utils import test_eval, rls_test_eval
from function_encoder.coefficients import recursive_least_squares_update

from models.maml import copy_model_params
from util.online_adapt import online_adapt_maml

from phoenix_util import *


def rls_loss_fn(model, batch, coeffs, device):
    xs, dt, ys, _, _, _ = batch
    xs = xs.to(device)
    dt = dt.to(device)
    ys = ys.to(device)
    pred = model((xs, dt), coefficients=coeffs)
    loss = torch.nn.functional.mse_loss(pred, ys)
    return loss

def function_encoder_loss_fn(model, batch, device):
    xs, dt, ys, example_xs, example_dt, example_ys = batch
    xs = xs.to(device)
    dt = dt.to(device)
    ys = ys.to(device)
    example_xs = example_xs.to(device)
    example_dt = example_dt.to(device)
    example_ys = example_ys.to(device)
    coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
    pred = model((xs, dt), coefficients=coefficients)
    loss = torch.nn.functional.mse_loss(pred, ys)
    return loss

def neural_ode_loss_fn(model, batch, device):
    xs, dt, ys, *_ = batch  # ignore example_xs, example_dt, example_ys if present
    xs = xs.to(device)
    dt = dt.to(device)
    ys = ys.to(device)
    pred = model((xs, dt))
    loss = torch.nn.functional.mse_loss(pred, ys)
    return loss

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_test_data(seed, eval_scene):
    # Load pre-split test data.
    load_path = f"Phoenix/data_split/seed_{seed}/scene_{eval_scene}"
    test_input = torch.tensor(pd.read_csv(f"{load_path}/test_input.csv", header=None).values).float()
    test_target = torch.tensor(pd.read_csv(f"{load_path}/test_target.csv", header=None).values).float()
    return TestDataset([test_input], [test_target], n_example_points=100)

def run_model_evaluation(model_types, batch, seed, device, n_basis):
    results = {}
    for model_type in model_types:
        path = f"logs/Phoenix_{model_type}/seed={seed}/{model_type}_model.pth"
        if not os.path.exists(path):
            print(f"Skipping missing: {path}")
            continue
        model = load_model(model_type, device, n_basis, path)
        if model_type == "function_encoder":
            loss_fn = function_encoder_loss_fn
        elif model_type == "neural_ode":
            loss_fn = neural_ode_loss_fn
            results[model_type] = test_eval(model, loss_fn, batch, device)
    return results

def plot_with_uncertainty(x_vals, data, label, color):
    # Plot the median and a shaded area between min and max.
    med = np.median(data, axis=0)
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    plt.plot(x_vals, med, label=label, color=color)
    plt.fill_between(x_vals, min_, max_, alpha=0.2, color=color, edgecolor="none", linewidth=0)


device = get_device()
n_basis = 8
init_scene = None
eval_scene = 1
max_steps = 100
# seeds = list(range(10))
seeds = [42]
model_types = ["neural_ode", "function_encoder"]

colors = {"neural_ode": "#D62728", "function_encoder": "#1F77B4", "rls": "#2ca02c"}
names = {"neural_ode": "Neural ODE", "function_encoder": "Function Encoder", "rls": "FE-RLS"}

all_results = {mt: np.zeros((len(seeds), max_steps)) for mt in model_types}
rls_results = np.zeros((len(seeds), max_steps))

for seed_idx, seed in enumerate(seeds):
    torch.manual_seed(seed)
    dataset = load_test_data(seed, eval_scene)

    # Initialize the RLS problem.
    model_path = f"logs/Phoenix_function_encoder/seed={seed}/function_encoder_model.pth"
    rls_model = load_model("function_encoder", device, n_basis, model_path)
    P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
    fixed_batch = tuple(x.to(device) for x in dataset[0][:-1])  # Exclude the last element (info)
    xs, dt, ys, *_ = fixed_batch

    if init_scene == None:
        coeffs = torch.zeros(1, n_basis, device=device)
    else:
        init_dataset = load_test_data(seed, init_scene)
        fixed_batch = tuple(x.to(device) for x in init_dataset[0][:-1])
        *_, ex_xs, ex_dt, ex_ys = fixed_batch
        init_x = ex_xs.unsqueeze(0)
        init_dt = ex_dt.unsqueeze(0)
        init_y = ex_ys.unsqueeze(0)
        coeffs, _ = rls_model.compute_coefficients((init_x, init_dt), init_y)


    for i in range(max_steps):
        # Get the next point in the RLS update data.
        x_step = xs[i, :6].unsqueeze(0).unsqueeze(0)
        u_step = xs[i, 6:].unsqueeze(0).unsqueeze(0)
        dt_step = dt[i].unsqueeze(0).unsqueeze(0)
        y_step = ys[i].unsqueeze(0).unsqueeze(0)

        # Compute new coeffs using RLS update. 
        g = rls_model.basis_functions((torch.cat((x_step, u_step), dim=-1), dt_step))
        L = torch.linalg.cholesky(P)
        coeffs, P = recursive_least_squares_update(
            method='qr', g=g, y=y_step, P=L, coefficients=coeffs, forgetting_factor=0.95
        )

        # Evaluate the FE-RLS model after RLS updates.
        eval_batch = tuple(x.to(device) for x in dataset[0][:-1])  # Exclude the last element (info)
        rls_results[seed_idx, i] = rls_test_eval(rls_model, rls_loss_fn, eval_batch, coeffs, device)

        # Evalute the baseline models. 
        seed_results = run_model_evaluation(model_types, eval_batch, seed, device, n_basis)
        for mt in model_types:
            if mt in seed_results:
                all_results[mt][seed_idx, i] = seed_results[mt]

# Plotting
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

fig = plt.figure(figsize=(3.5, 2.5))
x_vals = np.arange(max_steps)

for mt in model_types:
    if all_results[mt].size > 0:
        plot_with_uncertainty(x_vals, all_results[mt], names[mt], colors[mt])

plot_with_uncertainty(x_vals, rls_results, names['rls'], colors['rls'])

plt.yscale("log")
plt.xlabel("Number of Time Steps")
plt.ylabel("Mean Squared Error (MSE)")
fig.legend(loc="outside upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(f"rls_error_over_seeds_from_scene_{init_scene}_to_scene_{eval_scene}.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()