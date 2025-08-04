import os

from typing import List

import csv

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset

from .process_data import process_data


class PhoenixDataset(IterableDataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
        n_points: int,
    ):

        self.inputs = inputs  # list of tensors of shape [n_points, n_features]
        self.targets = targets  # list of tensors of shape [n_points, n_features]

        assert len(self.inputs) == len(
            self.targets
        ), "Inputs and targets must have the same length"

        for i in range(len(self.inputs)):
            if self.inputs[i].shape[0] != self.targets[i].shape[0]:
                raise ValueError(
                    f"Input and target tensors must have the same number of samples. "
                    f"Input shape: {self.inputs[i].shape}, Target shape: {self.targets[i].shape}"
                )

        self.n_example_points = n_example_points
        self.n_points = n_points
        self.xs_mean = None
        self.xs_std = None
        self.ys_mean = None
        self.ys_std = None

    def __iter__(self):
        while True:
            n_samples = self.n_points + self.n_example_points

            # Generate a random index
            B = torch.randint(0, len(self.inputs), (1,)).item()

            inputs = self.inputs[B]
            targets = self.targets[B]

            # Sample random points from the data without replacement
            indices = torch.randperm(inputs.shape[0])[:n_samples]
            _xs = inputs[indices, 1:]
            _dt = targets[indices, 0] - inputs[indices, 0]
            _ys = targets[indices, 1:] - _xs[:, :6]

            # Split the data
            example_xs = _xs[: self.n_example_points]
            example_dt = _dt[: self.n_example_points]
            example_ys = _ys[: self.n_example_points]

            xs = _xs[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            ys = _ys[self.n_example_points :]

            info = {}

            yield xs, dt, ys, example_xs, example_dt, example_ys, info


class TestDataset(Dataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
    ):
        self.inputs = inputs
        self.targets = targets
        self.n_example_points = n_example_points

        self.xs_mean = None
        self.xs_std = None
        self.ys_mean = None
        self.ys_std = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        # Sample random points from the data without replacement
        indices = torch.randperm(self.inputs[idx].shape[0])
        example_indices = indices[: self.n_example_points]
        query_indices = indices[self.n_example_points :]

        _xs = self.inputs[idx][:, 1:]
        _dt = self.targets[idx][:, 0] - self.inputs[idx][:, 0]
        _ys = self.targets[idx][:, 1:] - _xs[:, :6]
        example_xs = _xs[example_indices]
        example_dt = _dt[example_indices]
        example_ys = _ys[example_indices]
        xs = _xs[query_indices]
        dt = _dt[query_indices]
        ys = _ys[query_indices]

        info = {}
        return xs, dt, ys, example_xs, example_dt, example_ys, info


class OnlineTestDataset(Dataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
    ):
        self.inputs = inputs
        self.targets = targets
        self.n_example_points = n_example_points

    def __len__(self):
        return sum(
            t.shape[0] - self.n_example_points for t in self.targets
        )  # total number of samples across all scenes

    def __getitem__(self, idx):
        # get the correct index in the correct dataset
        lens = [t.shape[0] - self.n_example_points for t in self.targets]
        dataset = 0
        while(len(lens) > 0 and idx >= lens[0]):
            idx -= lens[0]
            lens = lens[1:]
            dataset += 1

        # get the correct dataset
        ins = self.inputs[dataset]
        tar = self.targets[dataset]


        # get this data
        _xs = ins[idx:idx + self.n_example_points + 1, 1:]
        _dt = (
            tar[idx:idx + self.n_example_points + 1, 0] 
            - ins[idx:idx + self.n_example_points + 1, 0]
        )
        _ys = tar[idx:idx + self.n_example_points + 1, 1:] - _xs[:, :6]
        example_xs = _xs[:self.n_example_points]
        example_dt = _dt[:self.n_example_points]
        example_ys = _ys[:self.n_example_points]
        xs = _xs[self.n_example_points:]
        dt = _dt[self.n_example_points:]
        ys = _ys[self.n_example_points:]

        assert example_xs.shape == (self.n_example_points, 8)
        assert example_dt.shape == (self.n_example_points,)
        assert example_ys.shape == (self.n_example_points, 6)
        assert xs.shape == (1, 8)
        assert dt.shape == (1,)
        assert ys.shape == (1, 6)

        info = {}
        return xs, dt, ys, example_xs, example_dt, example_ys, info

class MultiRolloutDataset(IterableDataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
        k_steps: int,
        n_rollouts: int,
    ):
        self.inputs = inputs
        self.targets = targets
        self.n_example_points = n_example_points
        self.k_steps = k_steps
        self.n_rollouts = n_rollouts

    def __iter__(self):
        while True:
            # Input/output tensors
            input_data = self.inputs[0]
            target_data = self.targets[0]
            total_window = self.k_steps + self.n_rollouts + 1  # +1 for final x0_n

            # Ensure we have enough space
            start_idx = torch.randint(input_data.shape[0] - total_window, (1,)).item()

            _xs = input_data[:, 1:]
            _dt = target_data[:, 0] - input_data[:, 0]
            _ys = target_data[:, 1:] - _xs[:, :6]

            # Random example points for coeffs init
            indices = torch.randperm(_xs.shape[0])
            example_indices = indices[:self.n_example_points]
            example_xs = _xs[example_indices]
            example_dt = _dt[example_indices]
            example_ys = _ys[example_indices]

            # Slice sequential windows for each rollout
            x0_seq = []     # initial state for each rollout
            dt_seq = []     # timestep per rollout
            u_seq = []      # control input per rollout
            y_seq = []      # target next state per rollout

            for n in range(self.n_rollouts):
                offset = start_idx + n

                # Initial state for this rollout
                x0_n = input_data[offset, 1:7]  # [pos, vel]
                x0_seq.append(x0_n)

                # Sequence of dt, u, y for this rollout
                dt_n = _dt[offset : offset + self.k_steps]
                u_n = _xs[offset : offset + self.k_steps, 6:]
                y_n = target_data[offset : offset + self.k_steps, 1:]

                dt_seq.append(dt_n)
                u_seq.append(u_n)
                y_seq.append(y_n)

            # Stack into tensors: shape [N, k, ...]
            x0_seq = torch.stack(x0_seq)          # [N, 6]
            dt_seq = torch.stack(dt_seq)          # [N, k]
            u_seq = torch.stack(u_seq)            # [N, k, 2]
            y_seq = torch.stack(y_seq)            # [N, k, 6]

            yield x0_seq, dt_seq, u_seq, y_seq, example_xs, example_dt, example_ys
            
def load_csv(filepath):
    """
    Load a CSV file and return the data as a tensor.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))  # path to terrain_adaptation/data
    repo_root = os.path.abspath(os.path.join(repo_root, "../.."))  # path to repo root
    full_path = os.path.join(repo_root, filepath)
    data = []
    with open(full_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)


def load_all_scenes():
    """
    Loads and processes odom and cmd_vel data for all scenes.
    Returns a dictionary: {scene_idx: (inputs, targets)}
    """

    data = {}
    for idx in range(8):
        odom = load_csv(f"Phoenix/data/scene{idx}_odom.csv")
        cmd_vel = load_csv(f"Phoenix/data/scene{idx}_cmd_vel.csv")
        inputs, targets = process_data(odom, cmd_vel)
        data[f"scene{idx}"] = (inputs, targets)
    # plot_target_data(data)
    return data

def load_scenes(scenes: List[int]):
    """
    Loads and processes odom and cmd_vel data for specified scenes.
    Returns a dictionary: {scene_idx: (inputs, targets)}
    """

    data = {}
    for idx in scenes:
        odom = load_csv(f"Phoenix/data/scene{idx}_odom.csv")
        cmd_vel = load_csv(f"Phoenix/data/scene{idx}_cmd_vel.csv")
        inputs, targets = process_data(odom, cmd_vel)
        data[f"scene{idx}"] = (inputs, targets)
    # plot_target_data(data)
    return data

def plot_target_data(data):
    """
    Plot the target data for all scenes.
    """
    for scene, (inputs, targets) in data.items():
        # Create scatter plots of the states in separate subfigures
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        fig.suptitle(f"{scene} Target Data")
        axs[0, 0].scatter(targets[:, 0], targets[:, 1], c="r", label="xPos")
        axs[0, 0].set_title("del_xPos")
        axs[0, 1].scatter(targets[:, 0], targets[:, 2], c="g", label="yPos")
        axs[0, 1].set_title("del_yPos")
        axs[1, 0].scatter(targets[:, 0], -inputs[:, 3] + targets[:, 3], c="b", label="yaw")
        axs[1, 0].set_title("del_yaw")
        axs[1, 1].scatter(targets[:, 0], -inputs[:, 4] + targets[:, 4], c="c", label="xVel")
        axs[1, 1].set_title("del_xVel")
        axs[2, 0].scatter(targets[:, 0], -inputs[:, 5] + targets[:, 5], c="m", label="yVel")
        axs[2, 0].set_title("del_yVel")
        axs[2, 1].scatter(targets[:, 0], -inputs[:, 6] + targets[:, 6], c="y", label="zAngVel")
        axs[2, 1].set_title("del_zAngVel")

        for ax in axs.flat:
            # ax.label_outer()
            ax.legend()
        print(torch.argmax(targets[:, 3]).item())
        plt.show()

    exit()