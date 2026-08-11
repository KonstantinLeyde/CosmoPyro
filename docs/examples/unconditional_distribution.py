import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from cosmopyro.distributions.grid_distributions import (
    InterpolatedConditional1D,
    normalize_cond_interpolated_1d,
)

# Define bin edges
x_edges = jnp.linspace(5.0, 100.0, 200)

# Some unnormalized density (e.g. from a parametrized model)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
raw_density = x_centers ** (-2.5)

# Normalize
density = normalize_cond_interpolated_1d(x_edges, raw_density)

# Wrap in a model
model = InterpolatedConditional1D(
    x_bins={"mass_1_s": x_edges},
    y_bins=None,
    cond=density,
)

# Evaluate
test_values = jnp.array([10.0, 30.0, 60.0])
log_p = model.log_prob(x_vals={"mass_1_s": test_values})
print("log_p:", log_p)
