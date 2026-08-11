import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp

from cosmopyro.distributions.grid_distributions import (
    InterpolatedConditional1D,
    normalize_cond_interpolated_1d,
)
from cosmopyro.distributions.mass_distribution_parametrized import (
    construct_prob_nn_multipeak_1D,
)
from cosmopyro.utils.jax_utils import compute_centers_and_delta_from_array

m1_edges = jnp.linspace(5.0, 100.0, 200)
m1_centers, m1_deltas = compute_centers_and_delta_from_array(m1_edges)

analysis = SimpleNamespace(
    binning=dict(
        boundaries=dict(mass_1_s=m1_edges),
        centers=dict(mass_1_s=m1_centers),
        deltas=dict(mass_1_s=m1_deltas),
    )
)

params = dict(
    mass_1_s=dict(
        alpha=3.5,
        mmin=5.0,
        mmax=87.0,
        lambda_g=0.04,
        lambda_g_low=0.5,
        delta_m=5.0,
        mu_g_low=10.0,
        sigma_g_low=2.0,
        mu_g_high=35.0,
        sigma_g_high=5.0,
    )
)

cond = construct_prob_nn_multipeak_1D(analysis, params)
cond = normalize_cond_interpolated_1d(m1_edges, cond)

model = InterpolatedConditional1D(
    x_bins={"mass_1_s": m1_edges},
    y_bins=None,
    cond=cond,
)

test_m = jnp.linspace(6.0, 85.0, 5000)
log_p = model.log_prob(x_vals={"mass_1_s": test_m})
area = jnp.trapezoid(jnp.exp(log_p), test_m)
print(f"Integral: {area:.4f}")
