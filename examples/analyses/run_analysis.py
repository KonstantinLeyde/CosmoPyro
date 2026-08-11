"""
Run a CosmoPyro analysis: load data, (optionally) run SVI, run MCMC.

Usage:
    python run_analysis.py \
        --job_id my_run \
        --path_kwargs path/to/kwargs_analysis.yaml \
        --path_posterior_samples data/posterior_samples.hdf5 \
        --path_injections data/injections.hdf5
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpyro

from cosmopyro.models.models import model_evaluate_p_theta
from cosmopyro.utils.analyses import Analysis
from cosmopyro.utils.runtime_utilities import get_config, get_time_stamp

# Parse command-line arguments
settings = get_config()

job_id = get_time_stamp("day").zfill(3) + "_" + settings["job_id"]

# 1. Set up analysis
analysis = Analysis(
    model=model_evaluate_p_theta,
    settings_path=settings.get("path_kwargs"),
    job_id=job_id,
)
analysis.update_kwargs_analysis(["run_kwargs"], settings)

# 2. Load data
data, skymap = analysis.load_data()

analysis.initialize_analysis()
analysis.set_data_kwargs(data=data, skymap=skymap)
analysis.run_data_checks()

# 3. (Optional) SVI initialization
init_strategy = None
guide_samples = None
best_sample = None
if analysis.use_svi_initialization:
    guide_samples, best_sample = analysis.run_svi_initialization(save_samples=True)
    init_strategy = numpyro.infer.initialization.init_to_value(values=best_sample)

not_dense_params = [
    "gaussian_F_whitened_spatial_white",
    "ratio_gaussian_whitened_field_white",
]

# 4. Run MCMC
if settings["sampler"] == "nuts":
    # Prefer the Laplace (Hessian) mass matrix when an SVI mode is available,
    # since SVI guide marginals systematically misjudge widths in tightly
    # constrained directions. Falls back to guide-marginal variances if no mode.
    kernel = analysis.configure_mcmc(
        init_strategy=init_strategy,
        not_dense_params=not_dense_params,
        guide_samples=guide_samples,
        mode_for_hessian=best_sample,
    )
    analysis.run_mcmc(kernel)
elif settings["sampler"] == "mclmc":
    analysis.run_mclmc(None, 1)
else:
    raise ValueError(f"Unknown sampler: {settings['sampler']}")
