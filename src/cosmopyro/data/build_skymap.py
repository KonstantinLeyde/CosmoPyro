import warnings
from types import SimpleNamespace

import jax.numpy as jnp

from .data_utils import save_namespace_to_hdf5

__all__ = [
    "build_skymap",
    "check_skymap_values",
    "validate_skymap",
]


def build_skymap(galaxy_catalog, discretization_3d, out_path=None, eps=1e-8):
    """
    Build a sky map from a galaxy catalog by histogramming into a 3D healpix grid.

    Parameters
    ----------
    galaxy_catalog : SimpleNamespace
        Must have .redshift, .ra, .dec. Optionally .host_log_prob as weights.
    discretization_3d : HealPixDiscretization3D
        Defines the redshift bins and healpix pixelization.
    out_path : str, optional
        If given, save to this HDF5 path.
    eps: float
        Small value to add to counts to avoid zero probabilities (if no weights).

    Returns
    -------
    SimpleNamespace with:
        .prob_skyposition_zhp : array (n_z, npix), normalized per redshift slice
        .z_edges : array (n_z + 1,)
        .nside : int
    """
    weights = None
    if hasattr(galaxy_catalog, "host_log_prob"):
        weights = jnp.exp(galaxy_catalog.host_log_prob)

    counts = discretization_3d.histogram_healpix_3d_from_r_dec_ra(
        galaxy_catalog.redshift,
        galaxy_catalog.dec,
        galaxy_catalog.ra,
        radial_key="r",
        weights_xyz=weights,
    )

    # Replace empty healpixels with eps so empty z-bins normalize to uniform
    counts = jnp.where(counts == 0, eps, counts)

    # Normalize each redshift slice to sum to 1 (uniform if empty)
    row_sums = jnp.sum(counts, axis=-1, keepdims=True)
    row_sums = jnp.where(row_sums > 0, row_sums, 1.0)
    prob_zhp = counts / row_sums

    skymap = SimpleNamespace(
        prob_skyposition_zhp=prob_zhp,
        z_edges=discretization_3d.boundaries["r"],
        nside=discretization_3d.nside,
    )

    if out_path is not None:
        save_namespace_to_hdf5(skymap, out_path)

    return skymap


def validate_skymap(skymap, binning, nside, atol=1e-6):
    """
    Validate that a sky map is compatible with the analysis binning.

    Parameters
    ----------
    skymap : SimpleNamespace
        Must have .prob_skyposition_zhp, .z_edges, .nside.
    binning : dict
        Analysis binning with binning['boundaries']['redshift'].
    nside : int
        Expected healpix nside.
    atol : float
        Tolerance for redshift edge comparison.

    Raises
    ------
    ValueError
        If the sky map doesn't match the analysis setup.
    """

    # Check healpix dimension
    expected_npix = 12 * nside**2
    if skymap.prob_skyposition_zhp.shape[-1] != expected_npix:
        raise ValueError(
            f"Sky map has {skymap.prob_skyposition_zhp.shape[-1]} pixels, "
            f"expected {expected_npix} for nside={nside}."
        )

    # Check redshift edges match
    z_edges_analysis = jnp.asarray(binning["boundaries"]["redshift"])
    z_edges_skymap = jnp.asarray(skymap.z_edges)

    if z_edges_analysis.shape[0] != z_edges_skymap.shape[0]:
        raise ValueError(
            f"Sky map has {z_edges_skymap.shape[0] - 1} redshift bins, "
            f"analysis expects {z_edges_analysis.shape[0] - 1}."
        )

    max_dev = float(jnp.max(jnp.abs(z_edges_analysis - z_edges_skymap)))
    if max_dev > atol:
        raise ValueError(
            f"Sky map redshift edges don't match analysis edges. "
            f"Max deviation: {max_dev:.2e} (tolerance: {atol:.2e})."
        )


def check_skymap_values(skymap):
    """
    Check that the sky map values are finite and non-negative; warn about exact zeros.

    Non-finite or negative entries are errors. Exact zeros are allowed but reported
    with a warning: the log-density there is floored at ``epsilon`` (gradient-safe,
    see ``grid_distributions._safe_log_density``), i.e. such cells are treated as
    (almost) impossible host locations, which is usually worth knowing.

    Parameters
    ----------
    skymap : SimpleNamespace
        Must have .prob_skyposition_zhp.

    Returns
    -------
    list of str
        Error messages; empty if the sky map passes.
    """
    prob = jnp.asarray(skymap.prob_skyposition_zhp)
    errors = []

    n_nonfinite = int(jnp.sum(~jnp.isfinite(prob)))
    if n_nonfinite > 0:
        errors.append(f"Sky map contains {n_nonfinite} non-finite (NaN/inf) entries.")

    finite = jnp.where(jnp.isfinite(prob), prob, 1.0)
    n_negative = int(jnp.sum(finite < 0))
    if n_negative > 0:
        errors.append(
            f"Sky map contains {n_negative} negative entries (min {float(jnp.min(finite)):.3e})."
        )

    is_zero = finite == 0
    n_zero = int(jnp.sum(is_zero))
    if n_zero > 0:
        zero_bins = jnp.nonzero(jnp.any(is_zero, axis=-1))[0]
        z_info = ""
        if hasattr(skymap, "z_edges"):
            z_edges = jnp.asarray(skymap.z_edges)
            z_info = (
                f" in {zero_bins.shape[0]} redshift bins, "
                f"z in [{float(z_edges[zero_bins[0]]):.4f}, {float(z_edges[zero_bins[-1] + 1]):.4f}]"
            )
        warnings.warn(
            f"Sky map contains {n_zero} entries ({n_zero / prob.size:.1e} of all) that are exactly "
            f"zero{z_info}. Posterior samples there get log(epsilon), i.e. these cells are "
            "treated as (almost) impossible host locations.",
            stacklevel=2,
        )

    return errors
