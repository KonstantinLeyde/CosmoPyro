import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from cosmopyro.distributions.grid_distributions import (
    InterpolatedConditional1D,
    normalize_cond_interpolated_1d,
)

# P(q | m1) with shape (n_q, n_m1)
q_edges = jnp.linspace(0.01, 1.0, 100)
m1_edges = jnp.linspace(5.0, 100.0, 80)
q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])

# Simple power-law conditional: q^1.5 for all m1
raw_cond = jnp.broadcast_to(
    q_centers[:, None] ** 1.5,
    (q_centers.shape[0], m1_edges.shape[0] - 1),
)

# Normalize over q for each m1 slice
cond = normalize_cond_interpolated_1d(q_edges, raw_cond)

model = InterpolatedConditional1D(
    x_bins={"mass_ratio": q_edges},
    y_bins={"mass_1_s": m1_edges},
    cond=cond,
    continuous_y_names=["mass_1_s"],
)

log_p = model.log_prob(
    x_vals={"mass_ratio": jnp.array([0.5])},
    y_vals={"mass_1_s": jnp.array([30.0])},
)
print("log_p:", log_p)
