"""
run_experiments.py

A single script to run both MAML_NODE and FE_NODE experiments
with minimal boilerplate. Tracks loss and compute time for each step.
"""
import time
import torch
import tqdm
from datasets.van_der_pol import (
    VanDerPolDataset,
    mu_piecewise_constant,
    mu_sinusoidal_modulation,
    mu_linear_ramp,
    mu_constant,
)
from models.get_model import get_model

# Import or define your per-algorithm step functions:
# Each should return a tuple (loss: torch.Tensor, output: Any)
from experiment_utils import (
    maml_inner_update_and_forward,
    fe_rls_update_and_forward,
)

# Dispatch table: only the pieces that vary per algorithm
algorithms = {
    "MAML_NODE": {
        "init":      lambda device: get_model("MAML_NODE", device),
        "load_dir": lambda model: model.load_state_dict(
            torch.load("./logs/VanDerPol_MAML_NODE/model.pth", map_location='cpu')
        ),
        "step":      maml_inner_update_and_forward,
        "metrics":   ("loss", "compute_time"),
    },
    "FE_NODE": {
        "init":      lambda device: get_model("FE_NODE", device),
        "load_dir": lambda model: model.load_state_dict(
            torch.load("./logs/VanDerPol_FE_NODE/model.pth", map_location='cpu')
        ),
        "step":      fe_rls_update_and_forward,
        "metrics":   ("loss", "compute_time"),
    },
}


def run_experiment(
    alg_name: str,
    device: str,
    dataset: VanDerPolDataset,
    mu_schedules: list,
    n_steps: int
) -> dict:
    """
    Run one algorithm over a set of mu schedules.

    Returns a dict of metric lists saved at ./logs/{alg_name}_metrics.pth
    """
    cfg   = algorithms[alg_name]
    model = cfg["init"](device)
    cfg["load_dir"](model)
    model.to(device).eval()

    # Prepare trackers
    logs = {metric: [] for metric in cfg["metrics"]}

    # Main loop
    for mu_fn in mu_schedules:
        for _ in tqdm.trange(n_steps, desc=f"{alg_name} @ {mu_fn.__name__}"):
            # Sample batch or trajectory
            data = dataset.sample(mu_fn)

            # Time the step
            t0 = time.perf_counter()
            loss, _ = cfg["step"](model, data)
            elapsed = time.perf_counter() - t0

            # Record
            logs["compute_time"].append(elapsed)
            logs["loss"].append(loss.item())

    # Save metrics
    save_path = f"./logs/{alg_name}_metrics.pth"
    torch.save(logs, save_path)
    print(f"Saved metrics for {alg_name} at {save_path}")
    return logs


if __name__ == "__main__":
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and schedules
    dataset = VanDerPolDataset(
        n_points=100,
        n_example_points=100,
        dt_range=(0.1, 0.1)
    )
    mu_schedules = [
        mu_piecewise_constant,
        mu_sinusoidal_modulation,
        mu_linear_ramp,
        mu_constant,
    ]

    # Number of steps per schedule
    n_steps = 5000

    # Run all algorithms
    for alg in algorithms:
        run_experiment(alg, device, dataset, mu_schedules, n_steps)  
