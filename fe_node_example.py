import torch
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
from datasets.van_der_pol import VanDerPolDataset, van_der_pol, mu_piecewise_constant, mu_linear_ramp, mu_sinusoidal_modulation
from architecture.neural_ode import rk4_step
from models.get_model import get_model
from util.coefficients import recursive_least_squares_update

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

# Load dataset
dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Load model
alg = 'FE_NODE'
model = get_model(algorithm=alg, device=device)
model.load_state_dict(torch.load(f"./logs/VanDerPol_{alg}/model.pth", map_location=device))
n_basis = len(model.basis_functions.basis_functions)

# Evaluate model
model.eval()

def qualitative_evaluation():
    """
    This function performs a qualitative evaluation of the model by rolling out the model
    on a batch of initial conditions and comparing it to the ground truth trajectory.
    """
    
    with torch.no_grad():

        batch = next(dataloader_iter)
        y0, dt, y1, y0_example, dt_example, y1_example, info = batch
        y0=y0.to(device)
        y1=y1.to(device)
        dt=dt.to(device)

        # Roll out the model on the batch of initial conditions
        # and compare to the ground truth trajectory
        s = 0.1 # simulation time step
        n = int(10 / s)  # number of steps to simulate

        ground_truth_traj = [y0.clone()]
        fe_node_traj = [y0.clone()]
        _dt = torch.tensor([s], device=device)  # time step for simulation
        coefficients = model.compute_coefficients(
            (y0, dt), y1
        )[0]  # coefficients for the basis functions
        for i in range(n):
            # ground truth trajectory
            current_state = ground_truth_traj[-1]
            change_in_state = rk4_step(van_der_pol, current_state, _dt, mu=info['mu'].to(device))
            resulting_state = current_state + change_in_state
            ground_truth_traj.append(resulting_state)

            fe_node_state = fe_node_traj[-1]
            # expand _dt to match the batch size and number of points
            expanded_dt = _dt.expand(fe_node_state.shape[0], -1).expand(-1, fe_node_state.shape[1])
            fe_node_dstate = model((fe_node_state, expanded_dt), coefficients=coefficients)
            fe_node_prediction = fe_node_state + fe_node_dstate
            fe_node_traj.append(fe_node_prediction)

        ground_truth_traj = torch.stack(ground_truth_traj, dim=0).detach().cpu().numpy()
        fe_node_traj = torch.stack(fe_node_traj, dim=0).detach().cpu().numpy()

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
                (_m,) = ax[i, j].plot(fe_node_traj[:, traj_idx, 0, 0], fe_node_traj[:, traj_idx, 0, 1], color='orange', label='FE_NODE')

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

def quantitative_evaluation(mu_function=mu_piecewise_constant, trange=5000):
    """
    This function performs a quantitative evaluation of the model by computing
    the mean squared error (MSE) between the model predictions and the ground truth.
    
    Args:
        mu_function (callable): A function that takes a time tensor and returns a time-varying mu tensor.
        trange (int): The range of time steps for evaluation.
    """
    with torch.no_grad():

        # Initialize the coefficients
        coefficients = torch.zeros(1, n_basis, device=device)
        P = torch.eye(n_basis, device=device).unsqueeze(0)

        # mu = torch.empty(1, device=device).uniform_(*dataset.mu_range)
        # mu = mu_function(t=torch.arange(trange, device=device), mu_range=dataset.mu_range, device=device)  # time-varying mu parameter
        mu = mu_function(t=torch.arange(trange, device=device), device=device)  # time-varying mu parameter
        # plotting_mu = [mu.item()]  # for plotting purposes, we will keep track of the mu parameter

        losses_baseline = []
        losses_rls = []
        coefficient_baseline_norms = []
        coefficient_rls_norms = []

        with tqdm.trange(trange) as tqdm_bar:
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

                # Compute the basis functions
                g = model.basis_functions((y0, dt))

                L = torch.linalg.cholesky(P)
                coefficients, P = recursive_least_squares_update(
                    method='qr',
                    g=g,
                    y=y1,
                    P=L,
                    coefficients=coefficients,
                    forgetting_factor=0.95,
                )

                # Generate a new batch of data for evaluation
                n_points = 1000
                _y0 = torch.empty(1, n_points, 2, device=device).uniform_(*dataset.y0_range)
                _dt = torch.empty(1, n_points, device=device).uniform_(*dataset.dt_range)
                _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu_t)

                n_example_points = 100
                y0_example = _y0[:, :n_example_points, :]
                dt_example = _dt[:, :n_example_points]
                y1_example = _y1[:, :n_example_points, :]
                y0 = _y0[:, n_example_points:, :]
                dt = _dt[:, n_example_points:]
                y1 = _y1[:, n_example_points:, :]

                # Compute the baseline error
                coefficients_baseline, _ = model.compute_coefficients(
                    (y0_example, dt_example), y1_example
                )
                pred_baseline = model((y0, dt), coefficients=coefficients_baseline)
                loss_baseline = torch.nn.functional.mse_loss(pred_baseline, y1)

                # Compute the recursive least squares prediction error
                pred = model((y0, dt), coefficients=coefficients)
                loss_rls = torch.nn.functional.mse_loss(pred, y1)

                losses_baseline.append(loss_baseline.item())
                losses_rls.append(loss_rls.item())

                coefficient_baseline_norms.append(
                    coefficients_baseline.norm(dim=-1).mean().item()
                )
                coefficient_rls_norms.append(coefficients.norm(dim=-1).mean().item())

                tqdm_bar.set_postfix(
                    {
                        "loss_baseline": f"{loss_baseline.item():.2e}",
                        "loss_rls": f"{loss_rls.item():.2e}",
                    }
                )

        mu_func_string = mu_function.__name__
        # Plot the losses
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(losses_baseline, label="Baseline")
        ax[0].plot(losses_rls, label="Recursive Least Squares")

        ax[1].plot(coefficient_baseline_norms, label="Baseline Coefficients Norm")
        ax[1].plot(coefficient_rls_norms, label="RLS Coefficients Norm")

        # log scale y axis
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

        # plt.legend()
        # plt.tight_layout()
        ax[0].set_title("Loss")
        ax[1].set_title("Coefficient Norms")

        # plt.show()
        fig.savefig(f"./logs/VanDerPol_{alg}/example_{mu_func_string}.png")

        # Plot the standalone loss
        fig, ax = plt.subplots(1, 1, figsize=(10,10))

        # plot mu as vertical lines for every 1000 steps
        plotting_mu = mu[torch.arange(trange) % 1000 == 0].detach().cpu().numpy().tolist()
        for i, m in enumerate(plotting_mu):
            x = i * 1000
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
        ax.minorticks_on()
        ax.grid(which="both", axis="y", linestyle=":", linewidth=0.5)
        ax.plot(losses_rls, label="FE NODE + RLS", color='C1')
        ax.plot(losses_baseline, label="Batch FE NODE", color='C0')

        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(f"./logs/VanDerPol_{alg}/losses_{mu_func_string}.png", bbox_inches='tight')

        # save the losses
        # losses_fe_node = torch.tensor(loss_rls, device=device)
        torch.save({
            "losses_fe_node": losses_rls,
        }, f"./logs/VanDerPol_{alg}/losses_{mu_func_string}.pth")

if __name__ == "__main__":
    qualitative_evaluation()
    quantitative_evaluation(mu_function=mu_piecewise_constant, trange=5000)
    quantitative_evaluation(mu_function=mu_sinusoidal_modulation, trange=5000)
    quantitative_evaluation(mu_function=mu_linear_ramp, trange=5000)
    print("Evaluation completed.")
