"""
Analyze CosmoPyro MCMC results: load posteriors, make corner plots
and mass distribution plots.

Usage:
    python analyze_results.py

Change the variables below to point to your results.
"""

import jax

jax.config.update("jax_enable_x64", True)

import os

from cosmopyro.models.models import model_evaluate_p_theta
from cosmopyro.utils.analyses import Result
from cosmopyro.utils.plotting import (
    make_mass_plot,
    plot_posterior_comparison,
)

# ============================================================
# >>> Change these to match your run <<<
# ============================================================
RESULTS_PATH = "results"
SETTINGS_PATH = os.path.join(RESULTS_PATH, "kwargs_analysis.yaml")
# ============================================================

# 1. Load result object
result = Result(model_evaluate_p_theta, settings_path=SETTINGS_PATH)

# 2. Load data (PE samples + injections) and sky map
data, skymap = result.load_data()
result.set_data_kwargs(data=data, analysis=result, skymap=skymap)

# 3. Load MCMC samples
result.load_inf_data(skip_tensors=False)
result.load_hypersamples(skip_diverging_samples=False)

# 4. Compute mass distributions on a grid
grid_ref, samples_postp = result.compute_main_results()

# 5. Corner plot of hyper-parameters
plots_dir = os.path.join(result.get_results_folder(), "plots")
os.makedirs(plots_dir, exist_ok=True)

posteriors = dict(
    cosmopyro=SimpleNamespace(
        label="HMC", posterior=result.hypersamples_df, color="green"
    )
)
if hasattr(result, "guide_hypersamples"):
    posteriors["Guide"] = SimpleNamespace(
        posterior=result.guide_hypersamples, label="CosmoPyro SVI", color="blue"
    )

plot_posterior_comparison(
    posteriors,
    filepath=os.path.join(plots_dir, "corner.png"),
)

filename = os.path.join(plots_dir, "mass_distribution_median.png")
make_mass_plot(grid_ref, samples_postp, idx="median", filename=filename)

result.make_diagnostic_plots()

print(f"Plots saved to {plots_dir}")
