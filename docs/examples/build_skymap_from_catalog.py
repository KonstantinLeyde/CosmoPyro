import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp

from cosmopyro.data.build_skymap import build_skymap, validate_skymap
from cosmopyro.utils.spherical_coordinates import HealPixDiscretization3D

# Define the discretization (must match your analysis config)
discretization = HealPixDiscretization3D(
    n_r=300,  # number of redshift bins
    r_min=0.0,  # minimum redshift
    r_max=2.0,  # maximum redshift
    nside=2,  # HEALPix nside (npix = 12 * nside^2 = 48)
)

# Example galaxy catalog
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
galaxy_catalog = SimpleNamespace(
    redshift=jax.random.uniform(k1, shape=(500,), minval=0.05, maxval=1.95),
    ra=jax.random.uniform(k2, shape=(500,), minval=0.0, maxval=2 * jnp.pi),
    dec=jax.random.uniform(k3, shape=(500,), minval=0.1, maxval=jnp.pi - 0.1),
)

# Build the sky map
skymap = build_skymap(galaxy_catalog, discretization)

print(f"prob shape: {skymap.prob_skyposition_zhp.shape}")
print(f"z_edges shape: {skymap.z_edges.shape}")
print(f"nside: {skymap.nside}")

# Validate against analysis binning
binning = {"boundaries": {"redshift": discretization.boundaries["r"]}}
validate_skymap(skymap, binning, nside=2)
print("Validation passed.")
