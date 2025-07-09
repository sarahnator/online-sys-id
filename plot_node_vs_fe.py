import torch
import matplotlib.pyplot as plt


tests = "mu_constant mu_linear_ramp mu_sinusoidal_modulation mu_piecewise_constant".split()

for test in tests:
    losses_fe = torch.load(f"./logs/VanDerPol_FE_NODE/losses_{test}.pth")
    losses_fe_rls = losses_fe["losses_fe_rls"]
    losses_fe_baseline = losses_fe["losses_fe_baseline"]
    losses_maml = torch.load(f"./logs/VanDerPol_MAML2_NODE/losses_{test}.pth")["losses_maml"].cpu()
    # losses_maml = torch.load(f"./logs/VanDerPol_NODE/losses_{test}.pth")["losses_maml"].cpu()

    # Plot the losses
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(losses_fe_baseline, label="Batch NODE-FE", color='C0')
    ax.plot(losses_fe_rls, label="NODE-FE + RLS", color='C1')
    ax.plot(losses_maml, label="NODE + MAML", color='C2')

    # log scale y axis
    ax.set_yscale("log")

    plt.legend()
    plt.tight_layout()


    # plt.show()
    fig.savefig(f"./logs/VanDerPol_comparisons/{test}.png")

    # # Plot the standalone loss
    # fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # # plot mu as vertical lines for every 1000 steps
    # plotting_mu = mu[torch.arange(trange) % 1000 == 0].detach().cpu().numpy().tolist()
    # for i, m in enumerate(plotting_mu):
    #     x = i * 1000
    #     ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    #     ax.text(
    #         x,               # data x
    #         0.1,               # axis-fraction y = 0 (bottom of the plotting area)
    #         f"$\\mu$={m:.1f}",
    #         transform=ax.get_xaxis_transform(),  # <-- key!
    #         rotation=90,
    #         va='bottom',     # push the text upward from the axis spine
    #         ha='left',
    #         fontsize=11,
    #         fontweight='bold',
    #     )

    # ax.set_yscale("log")
    # ax.minorticks_on()
    # ax.grid(which="both", axis="y", linestyle=":", linewidth=0.5)
    # ax.plot(losses_rls, label="FE NODE + RLS", color='C1')
    # ax.plot(losses_baseline, label="Batch FE NODE", color='C0')

    # plt.legend()
    # plt.tight_layout()
    # # plt.show()
    # fig.savefig(f"./logs/VanDerPol_{alg}/losses_{mu_func_string}.png", bbox_inches='tight')
