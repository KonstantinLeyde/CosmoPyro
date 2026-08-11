import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp

from cosmopyro.distributions.mass_distribution_parametrized import (
    get_log_window_mass_s,
)
from cosmopyro.distributions.mass_distributions_gaussian_process import (
    construct_log_prob_nn_whitened_field_1D,
)
from cosmopyro.utils.jax_utils import compute_centers_and_delta_from_array

# Set up binning
n_bins = 200
m1_edges = jnp.linspace(2.0, 120.0, n_bins + 1)
m1_centers, m1_deltas = compute_centers_and_delta_from_array(m1_edges)

analysis = SimpleNamespace(
    binning=dict(
        boundaries=dict(mass_1_s=m1_edges),
        centers=dict(mass_1_s=m1_centers),
        deltas=dict(mass_1_s=m1_deltas),
    )
)

# Draw multiple realizations
for i in range(3):
    key = jax.random.PRNGKey(i)
    noise = jax.random.normal(key, shape=(n_bins,))

    params = dict(
        mass_1_s=dict(
            gaussian_F_whitened_spatial=noise,
            mass_min=5.0,
            mass_max=80.0,
            sigma_low_fractional=0.05,
            sigma_high_fractional=0.05,
            power_spectrum_amplitude=5.0,
            power_spectrum_cutoff=5.0,
        )
    )

    log_window = get_log_window_mass_s(analysis, params)
    log_prob = construct_log_prob_nn_whitened_field_1D(analysis, params, log_window)
    prob = jnp.exp(log_prob)
    integral = jnp.sum(prob * m1_deltas)
    print(f"Draw {i}: shape={prob.shape}, integral={integral:.4f}")
