import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp

from cosmopyro.data.build_skymap import validate_skymap

nside = 2
npix = 12 * nside**2
z_edges = jnp.linspace(0.0, 2.0, 301)
n_z = z_edges.shape[0] - 1

# Your custom density array (e.g. from galaxy number counts)
prob_zhp = jnp.ones((n_z, npix)) / npix  # uniform example

skymap = SimpleNamespace(
    prob_skyposition_zhp=prob_zhp,
    z_edges=z_edges,
    nside=nside,
)

# Validate
binning = {"boundaries": {"redshift": z_edges}}
validate_skymap(skymap, binning, nside=nside)
print(f"Custom skymap: shape={prob_zhp.shape}, nside={nside}")
