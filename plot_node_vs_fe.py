import torch
import matplotlib.pyplot as plt


tests = "mu_constant mu_linear_ramp mu_sinusoidal_modulation mu_piecewise_constant".split()

for test in tests:
    losses_fe = torch.load(f"./logs/VanDerPol_FE_NODE/losses_{test}.pth")
    losses_fe_rls = losses_fe["losses_fe_rls"]
    losses_fe_baseline = losses_fe["losses_fe_baseline"]
    mu = losses_fe["mu"]
    # losses_maml = torch.load(f"./logs/VanDerPol_MAML2_NODE/losses_{test}.pth")["losses_maml"].cpu()
    losses_maml_50_shot = torch.load(f"./logs/VanDerPol_NODE/losses_{test}_50.pth")["losses_node_window"].cpu()
    losses_maml_100_shot = torch.load(f"./logs/VanDerPol_NODE/losses_{test}_100.pth")["losses_node_window"].cpu()
    losses_maml_one_shot = torch.load(f"./logs/VanDerPol_NODE/losses_{test}_50.pth")["losses_node"].cpu()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # plot mu as vertical lines for every 1000 steps
    plotting_mu = mu[torch.arange(len(mu)) % 1000 == 0].detach().cpu().numpy().tolist()
    for i, m in enumerate(plotting_mu):
        x = i * 1000
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        ax.text(
            x,               # data x
            0.1,               # axis-fraction y = 0 (bottom of the plotting area)
            f"$\\mu$={m:.2f}",
            transform=ax.get_xaxis_transform(),  # <-- key!
            rotation=90,
            va='bottom',     # push the text upward from the axis spine
            ha='left',
            fontsize=11,
            fontweight='bold',
        )

    # Plot the losses - maml first
    ax.plot(losses_maml_one_shot, label="NODE + MAML (1-shot)", color='C4')
    ax.plot(losses_maml_50_shot, label="NODE + MAML (50-shot)", color='C2')
    ax.plot(losses_maml_100_shot, label="NODE + MAML (100-shot)", color='C3')



    # log scale y axis
    ax.set_yscale("log")
    plt.legend()
    plt.tight_layout()


    # plt.show()
    fig.savefig(f"./logs/VanDerPol_comparisons/maml_{test}.png")

    ax.plot(losses_fe_baseline, label="Batch NODE-FE", color='C0')
    ax.plot(losses_fe_rls, label="NODE-FE + RLS", color='C1')
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"./logs/VanDerPol_comparisons/{test}.png")


    # # Plot the standalone loss
    # fig, ax = plt.subplots(1, 1, figsize=(10,10))


    # ax.set_yscale("log")
    # ax.minorticks_on()
    # ax.grid(which="both", axis="y", linestyle=":", linewidth=0.5)
    # ax.plot(losses_rls, label="FE NODE + RLS", color='C1')
    # ax.plot(losses_baseline, label="Batch FE NODE", color='C0')

    # plt.legend()
    # plt.tight_layout()
    # # plt.show()
    # fig.savefig(f"./logs/VanDerPol_{alg}/losses_{test}.png", bbox_inches='tight')
