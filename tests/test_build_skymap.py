from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from cosmopyro.data.build_skymap import build_skymap, validate_skymap
from cosmopyro.utils.spherical_coordinates import HealPixDiscretization3D


@pytest.fixture
def discretization():
    return HealPixDiscretization3D(n_r=5, r_min=0.0, r_max=1.0, nside=2)


@pytest.fixture
def galaxy_catalog():
    """Small catalog with galaxies at known positions."""
    n = 500
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    return SimpleNamespace(
        redshift=jax.random.uniform(k1, shape=(n,), minval=0.05, maxval=0.95),
        dec=jax.random.uniform(k2, shape=(n,), minval=0.1, maxval=jnp.pi - 0.1),
        ra=jax.random.uniform(k3, shape=(n,), minval=0.0, maxval=2 * jnp.pi),
    )


def test_build_skymap_shape(discretization, galaxy_catalog):
    skymap = build_skymap(galaxy_catalog, discretization)

    assert skymap.prob_skyposition_zhp.shape == discretization.shape
    assert skymap.z_edges.shape[0] == discretization.n_r + 1
    assert skymap.nside == discretization.nside


def test_build_skymap_normalized(discretization, galaxy_catalog):
    skymap = build_skymap(galaxy_catalog, discretization)

    row_sums = jnp.sum(skymap.prob_skyposition_zhp, axis=-1)
    # Each redshift slice with galaxies should sum to 1
    nonempty = row_sums > 0
    assert jnp.allclose(row_sums[nonempty], 1.0, atol=1e-10)


def test_build_skymap_non_negative(discretization, galaxy_catalog):
    skymap = build_skymap(galaxy_catalog, discretization)
    assert jnp.all(skymap.prob_skyposition_zhp >= 0)


def test_build_skymap_with_weights(discretization, galaxy_catalog):
    galaxy_catalog.host_log_prob = jnp.zeros(galaxy_catalog.redshift.shape[0])
    skymap = build_skymap(galaxy_catalog, discretization)

    assert skymap.prob_skyposition_zhp.shape == discretization.shape
    row_sums = jnp.sum(skymap.prob_skyposition_zhp, axis=-1)
    nonempty = row_sums > 0
    assert jnp.allclose(row_sums[nonempty], 1.0, atol=1e-10)


def test_build_skymap_empty_slice():
    """Redshift bins with no galaxies should not cause NaN."""
    disc = HealPixDiscretization3D(n_r=3, r_min=0.0, r_max=2.0, nside=1)
    # All galaxies in a narrow redshift band
    catalog = SimpleNamespace(
        redshift=jnp.array([0.5, 0.5, 0.5]),
        dec=jnp.array([1.0, 1.0, 1.0]),
        ra=jnp.array([0.5, 1.5, 2.5]),
    )
    counts_should_be_z = jnp.histogram(catalog.redshift, bins=disc.boundaries["r"])[0]

    eps = 0.0
    skymap = build_skymap(catalog, disc, eps=eps)

    assert not jnp.any(jnp.isnan(skymap.prob_skyposition_zhp))
    # Empty slices should be all zeros

    prob_z = skymap.prob_skyposition_zhp.sum(axis=-1)

    for i in range(disc.n_r):
        if counts_should_be_z[i] == 0:
            assert jnp.all(prob_z[i] == 0)
        else:
            assert jnp.all(prob_z[i] > 0)


# --- validate_skymap tests ---


def test_validate_skymap_passes(discretization, galaxy_catalog):
    skymap = build_skymap(galaxy_catalog, discretization)
    binning = {"boundaries": {"redshift": discretization.boundaries["r"]}}
    validate_skymap(skymap, binning, nside=discretization.nside)


def test_validate_skymap_wrong_nside(discretization, galaxy_catalog):
    skymap = build_skymap(galaxy_catalog, discretization)
    binning = {"boundaries": {"redshift": discretization.boundaries["r"]}}
    with pytest.raises(ValueError, match="nside"):
        validate_skymap(skymap, binning, nside=4)


def test_validate_skymap_wrong_n_redshift_bins(discretization, galaxy_catalog):
    skymap = build_skymap(galaxy_catalog, discretization)
    wrong_edges = jnp.linspace(0.0, 1.0, 20)
    binning = {"boundaries": {"redshift": wrong_edges}}
    with pytest.raises(ValueError, match="redshift bins"):
        validate_skymap(skymap, binning, nside=discretization.nside)


def test_validate_skymap_misaligned_edges(discretization, galaxy_catalog):
    skymap = build_skymap(galaxy_catalog, discretization)
    shifted_edges = discretization.boundaries["r"] + 0.1
    binning = {"boundaries": {"redshift": shifted_edges}}
    with pytest.raises(ValueError, match="don't match"):
        validate_skymap(skymap, binning, nside=discretization.nside)
