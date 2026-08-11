"""
Loading and inspecting CosmoPyro results.
Shows how to load chains, plot posteriors, and visualize mass distributions.
"""

import jax

jax.config.update("jax_enable_x64", True)

import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cosmopyro.models import models
from cosmopyro.utils import analyses
from cosmopyro.utils.plotting import make_mass_plot

# --- Step 1: Load results ---
SETTINGS_PATH = "../../results/YOUR_JOB_ID/kwargs_analysis.yaml"  # <-- CHANGE THIS

result = analyses.Result(
    models.model_evaluate_p_theta,
    settings_path=SETTINGS_PATH,
)

# Load MCMC chains (skip large tensor parameters for speed)
result.load_inf_data(skip_tensors=True)
result.load_hypersamples(skip_diverging_samples=True)

print(f"Loaded {result.hypersamples_df.shape[0]} posterior samples")
print(f"Parameters: {list(result.hypersamples_df.columns)}")

# --- Step 2: Corner plot ---
fig = corner.corner(
    result.hypersamples_df,
    color="green",
    plot_datapoints=False,
    plot_contours=True,
    no_fill_contours=True,
    smooth=0.05,
    levels=[1 - jnp.exp(-(r**2) / 2) for r in [0.5, 1, 2]],
)
plt.savefig("corner.png", dpi=150)
print("Saved corner.png")

# --- Step 3: Mass distribution posterior predictive ---
# Reload with tensors to compute mass distributions
result_full = analyses.Result(
    models.model_evaluate_p_theta,
    settings_path=SETTINGS_PATH,
)
result_full.load_inf_data(skip_tensors=False)
result_full.load_hypersamples(skip_diverging_samples=True)

data, skymap = result_full.load_data()
result_full.set_data_kwargs(data=data, skymap=skymap)

grid_ref, samples_postp = result_full.compute_main_results(batch_size=200)

# Plot mass distribution for first 3 posterior draws
for i in range(3):
    make_mass_plot(grid_ref, samples_postp, i, f"mass_distribution_{i}.png")
    print(f"Saved mass_distribution_{i}.png")

# --- Step 4: Compare with true population (if available) ---
# population_ref = load_hdf5_to_namespace("../../data/delta_catalogs/catalog_REF.hdf5")
# plt.hist(population_ref.mass_1_s, bins=100, density=True, histtype='step',
#          color='green', linestyle='--', label='True population')
