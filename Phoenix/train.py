import csv
import os

import torch

from matplotlib import pyplot as plt
import numpy as np

from data.load_data import load_scenes, PhoenixDataset, TestDataset

import tqdm

from train_utils import train_step, test_eval
import argparse


# Load a CSV file and return the data as a tensor.
def load_csv(filepath):
    data = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)


# Parse command line arguments.
args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=42)
# args.add_argument("--model", type=str, default="maml2_node")
args.add_argument("--model", type=str, default="maml_node")
args.add_argument("--n_basis", type=int, default=8)
args.add_argument("--gradsteps", type=int, default=10000)
args.add_argument("--logdir", type=str, default="logs")
args.add_argument("--device", type=str, default="cuda")
args = args.parse_args()

# Create a personal logdir.
args.logdir = f"{args.logdir}/Phoenix_{args.model}/seed={args.seed}"

# Seed the model.
torch.manual_seed(args.seed)

# Choose the scenes for training, interpolation, and extrapolation.
training_scenes = [0, 2, 3, 4, 6, 7]
interpolation_scenes = [5]
extrapolation_scenes = [1]
all_scenes = training_scenes + interpolation_scenes + extrapolation_scenes

# Load interpolation and extrapoliation data as a dictionary.
scene_data = load_scenes(interpolation_scenes + extrapolation_scenes)

# Define lists to hold data for all scenes.
train_inputs = []
train_targets = []
unseen_test_inputs = []
unseen_test_targets = []
interpolation_inputs, interpolation_targets = [], []
extrapolation_inputs, extrapolation_targets = [], []

# Iterate over the train scenes and load the pre-split data.
for scene_index in training_scenes:

    # Load the data for the current scene.
    load_path = f"Phoenix/data_split/seed_{args.seed}/scene_{scene_index}"
    train_input = torch.tensor(load_csv(f"{load_path}/train_input.csv")).float()
    train_target = torch.tensor(load_csv(f"{load_path}/train_target.csv")).float()
    test_input = torch.tensor(load_csv(f"{load_path}/test_input.csv")).float()
    test_target = torch.tensor(load_csv(f"{load_path}/test_target.csv")).float()

    # Save the train and test data for training scenes.
    train_inputs.append(train_input)
    train_targets.append(train_target)
    unseen_test_inputs.append(test_input)
    unseen_test_targets.append(test_target)

# Iterate over the interpolation scenes. Data is not shuffled. 
for scene_index in interpolation_scenes:
    scene_str = f"scene{scene_index}"
    scene_input, scene_target = scene_data[scene_str]
    interpolation_inputs.append(scene_input)
    interpolation_targets.append(scene_target)

# Iterate over the extrapolation scenes. Data is not shuffled.
for scene_index in extrapolation_scenes:
    scene_str = f"scene{scene_index}"
    scene_input, scene_target = scene_data[scene_str]
    extrapolation_inputs.append(scene_input)
    extrapolation_targets.append(scene_target)

# create an iterable dataset for training
train_dataset = PhoenixDataset(
    train_inputs, train_targets, n_example_points=100, n_points=1000
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
train_dataloader_iter = iter(train_dataloader)

# create datasets for eval
eval_dataset = TestDataset(
    unseen_test_inputs, unseen_test_targets, n_example_points=100
)
interpolation_dataset = TestDataset(
    interpolation_inputs, interpolation_targets, n_example_points=100
)
extrapolation_dataset = TestDataset(
    extrapolation_inputs, extrapolation_targets, n_example_points=100
)

# Define model
from models.get_model import *

match args.model:
    case "neural_ode":
        model = NeuralODE(
            ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
            integrator=rk4_step,
        ).to(args.device)
        model.loss_fn = neural_ode_loss
        # loss_fn = neural_ode_loss
    case "function_encoder":
        model = FunctionEncoder(basis_functions=BasisFunctions(
                        *[
                            NeuralODE(
                                ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
                                integrator=rk4_step,
                            )
                            for _ in range(args.n_basis)
                        ]
                    )).to(args.device)
        model.loss_fn = neural_ode_loss
        # loss_fn = neural_ode_loss
    case "maml2_node":
        model = MAML2_NODE(
            NeuralODE(ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
            integrator=rk4_step),
            meta_learning_rate=1e-3,
            internal_learning_rate=1e-3
        ).to(args.device)

        model.loss_fn = neural_ode_loss
        # loss_fn = neural_ode_loss
    case "maml_node":
        model = MAML_NODE(
            NeuralODE(ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
            integrator=rk4_step),
            meta_learning_rate=1e-3,
            internal_learning_rate=1e-3,
            window_size=100,  # Example window size, adjust as needed
        ).to(args.device)
        model.loss_fn = neural_ode_loss
        # loss_fn = neural_ode_loss
    case _:
        raise ValueError(f"Unknown model type: {args.model_type}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
training_data = {
    "training_loss": [],
    "eval_loss": [],
    "interpolation_loss": [],
    "extrapolation_loss": [],
}

loss_fn = model.loss_fn
with tqdm.trange(args.gradsteps) as tqdm_bar:
    for grad_step in tqdm_bar:
        # Get the next batch of data.
        batch = next(train_dataloader_iter)
        train_loss = train_step(model, optimizer, loss_fn, batch, args.device)
        training_data["training_loss"].append(train_loss)

        # Evaluate on all eval datasets
        test_loss = 0
        for batch in eval_dataset:
            test_loss += test_eval(model, loss_fn, batch, args.device)
        test_loss /= len(eval_dataset)
        training_data["eval_loss"].append(test_loss)
        tqdm_bar.set_postfix_str(f"loss: {test_loss:.2e}")

        # evaluate on interpolation dataset
        interpolation_loss = 0
        for batch in interpolation_dataset:
            interpolation_loss += test_eval(model, loss_fn, batch, args.device)
        interpolation_loss /= len(interpolation_dataset)
        training_data["interpolation_loss"].append(interpolation_loss)

        # evaluate on extrapolation dataset
        extrapolation_loss = 0
        for batch in extrapolation_dataset:
            extrapolation_loss += test_eval(model, loss_fn, batch, args.device)
        extrapolation_loss /= len(extrapolation_dataset)
        training_data["extrapolation_loss"].append(extrapolation_loss)

# Save the model and data
os.makedirs(args.logdir, exist_ok=True)
torch.save(training_data, f"{args.logdir}/training_data.pth")

torch.save(model.state_dict(), f"{args.logdir}/{args.model}_model.pth")

# create a training curve plot just to visualize
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(training_data["training_loss"])
axs[0, 0].set_title("Training Loss")
axs[0, 1].plot(training_data["eval_loss"])
axs[0, 1].set_title("Eval Loss")
axs[1, 0].plot(training_data["interpolation_loss"])
axs[1, 0].set_title("Interpolation Loss")
axs[1, 1].plot(training_data["extrapolation_loss"])
axs[1, 1].set_title("Extrapolation Loss")
plt.tight_layout()
plt.savefig(f"{args.logdir}/training_curve.png")
