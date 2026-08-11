import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cosmopyro.distributions.grid_distributions import (
    InterpolatedConditional1D,
)
from cosmopyro.distributions.mass_distributions_gaussian_process import (
    construct_prob_nn_whitened_field_2D_logMdelta,
)
from cosmopyro.distributions.source_frame_masses import (
    construct_conditionals_from_prob_logM_delta,
)
from cosmopyro.utils.jax_utils import compute_centers_and_delta_from_array

# Set up binning
n_logM, n_delta = 120, 120
logM_edges = jnp.linspace(1.5, 6.0, n_logM + 1)
delta_edges = jnp.linspace(0.0, 4.0, n_delta + 1)

logM_centers, logM_deltas = compute_centers_and_delta_from_array(logM_edges)
delta_centers, delta_deltas = compute_centers_and_delta_from_array(delta_edges)

analysis = SimpleNamespace(
    binning=dict(
        boundaries=dict(log_mass_total_s=logM_edges, minus_log_mass_ratio=delta_edges),
        centers=dict(log_mass_total_s=logM_centers, minus_log_mass_ratio=delta_centers),
        deltas=dict(log_mass_total_s=logM_deltas, minus_log_mass_ratio=delta_deltas),
    )
)

# (m1_s, q) grid for visualization
bins_m1s = jnp.linspace(3.0, 120.0, 200)
bins_q = jnp.linspace(0.02, 1.0, 200)
bins_m2s = bins_m1s[:, None] * bins_q[None, :]
bins_logM = jnp.log(bins_m1s[:, None] + bins_m2s)
bins_delta = -jnp.log(bins_q[None, :]) * jnp.ones_like(bins_m1s[:, None])

# Draw 6 realizations
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for i, ax in enumerate(axes.flat):
    key = jax.random.PRNGKey(i)
    noise = jax.random.normal(key, shape=(n_logM, n_delta))

    params = dict(
        source_frame_masses=dict(
            gaussian_F_whitened_spatial=noise,
            mass_min=5.0,
            mass_max=80.0,
            sigma_low_fractional=0.05,
            sigma_high_fractional=0.05,
            power_spectrum_amplitude=0.045,
            power_spectrum_cutoff=50.0,
            power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio=1.0,
            power_law_reference_mass_1_s=-2.0,
            power_law_reference_mass_ratio=1.5,
        )
    )

    prob_logM_delta = construct_prob_nn_whitened_field_2D_logMdelta(analysis, params)

    # Factorize into p(logM) and p(delta | logM)
    prob_logM, prob_delta_given_logM = construct_conditionals_from_prob_logM_delta(
        analysis, prob_logM_delta
    )

    model_logM = InterpolatedConditional1D(
        x_bins={"log_mass_total_s": logM_edges},
        y_bins=None,
        cond=prob_logM,
    )
    model_delta = InterpolatedConditional1D(
        x_bins={"minus_log_mass_ratio": delta_edges},
        y_bins={"log_mass_total_s": logM_edges},
        cond=prob_delta_given_logM,
        continuous_y_names=["log_mass_total_s"],
    )

    # Evaluate on (m1_s, q) grid
    log_p_logM = model_logM.log_prob({"log_mass_total_s": bins_logM})
    log_p_delta = model_delta.log_prob(
        {"minus_log_mass_ratio": bins_delta},
        y_vals={"log_mass_total_s": bins_logM},
    )
    log_p = log_p_logM + log_p_delta

    ax.pcolormesh(
        bins_m1s,
        bins_q,
        jnp.exp(log_p).T,
        shading="auto",
        cmap="viridis",
    )
    ax.set_xlabel(r"$m_{1,s}$")
    ax.set_ylabel(r"$q$")
    ax.set_title(f"Prior draw {i + 1}")

plt.tight_layout()
