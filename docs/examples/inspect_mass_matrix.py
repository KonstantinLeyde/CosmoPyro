"""Visualize the NUTS mass matrix from a saved MCMC state."""

import matplotlib.pyplot as plt
import numpy as np

from cosmopyro.utils.helper_functions import last_state_read

# --- Load state ---
RESULTS_PATH = "/path/to/results/id_X/"
state = last_state_read(RESULTS_PATH + "preliminary/mcmc_last_state.pkl")

adapt = state.adapt_state
print("Step size:", adapt.step_size)
print("HMC sites:", list(state.z.keys()))

# inverse_mass_matrix is a dict: {(param_tuple,): array}
for key, inv_M in adapt.inverse_mass_matrix.items():
    param_names = list(key)
    inv_M = np.asarray(inv_M)
    print(f"\nBlock: {param_names}")
    print(f"  Shape: {inv_M.shape}")

    if inv_M.ndim == 1:
        # Diagonal mass matrix
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(len(inv_M)), inv_M)
        ax.set_xticks(range(len(inv_M)))
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylabel("Inverse mass (diagonal)")
        ax.set_title("Diagonal inverse mass matrix")

    elif inv_M.ndim == 2:
        # Dense mass matrix — show as correlation matrix
        std = np.sqrt(np.diag(inv_M))
        std_safe = np.where(std > 0, std, 1.0)
        corr = inv_M / np.outer(std_safe, std_safe)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Inverse mass matrix
        im1 = ax1.imshow(np.log10(np.abs(inv_M) + 1e-30), aspect="auto")
        ax1.set_xticks(range(len(param_names)))
        ax1.set_yticks(range(len(param_names)))
        ax1.set_xticklabels(param_names, rotation=45, ha="right", fontsize=7)
        ax1.set_yticklabels(param_names, fontsize=7)
        ax1.set_title("log10 |Inverse mass matrix|")
        fig.colorbar(im1, ax=ax1, shrink=0.8)

        # Correlation structure
        im2 = ax2.imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(param_names)))
        ax2.set_yticks(range(len(param_names)))
        ax2.set_xticklabels(param_names, rotation=45, ha="right", fontsize=7)
        ax2.set_yticklabels(param_names, fontsize=7)
        ax2.set_title("Correlation structure")
        fig.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "plots/mass_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
