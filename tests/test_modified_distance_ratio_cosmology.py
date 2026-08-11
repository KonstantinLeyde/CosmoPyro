# test_modified_distance_ratio_cosmology.py
import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from astropy.cosmology import FlatLambdaCDM

from cosmopyro.cosmology.modified_gw_distance_ratio import (
    ModifiedGWDistanceFlatLambdaCDM,
    complete_params_modified_gravity,
)

# ---------- Base (unchanged) fixtures ----------


@pytest.fixture(scope="module")
def params():
    return dict(H0=70.0, Omega_m=0.3)


@pytest.fixture(scope="module")
def numerics():
    return dict(z_min=1e-5, z_max=3.0, z_steps=200_000)


@pytest.fixture(scope="module")
def z_vals():
    return jnp.linspace(1e-5, 1.0, 100_000)


# ---------- Parametrized ratio model specs ----------


@pytest.fixture(
    scope="module",
    params=[
        dict(
            name_modified_ratio="FlatLambdaCDM_GW_distance_cosine",
            make_params_ratio=lambda: (
                lambda key0, key1: dict(
                    alphas=jax.random.normal(key0, (3,)) * 0.04,
                    phases=jax.random.uniform(key1, (3,)) * 2 * jnp.pi,
                    zmax_b=3.0,
                    z_tr=0.5,
                    z_max=3.0,
                )
            )(jax.random.key(0), jax.random.key(1)),
        ),
        dict(
            name_modified_ratio="FlatLambdaCDM_GW_distance_gp_integrated",
            make_params_ratio=lambda: (
                lambda key: dict(
                    ratio_power_spectrum_amplitude=0.001,
                    ratio_amplitude_at_z_max=1.2,
                    ratio_gaussian_whitened_field=jax.random.normal(key, (200,)),
                )
            )(jax.random.key(2)),
        ),
    ],
    ids=["FlatLambdaCDM_GW_distance_cosine", "FlatLambdaCDM_GW_distance_gp_integrated"],
)
def ratio_spec(request):
    name = request.param["name_modified_ratio"]
    params_ratio = request.param["make_params_ratio"]()

    kwargs = dict(
        cosmology_model_name=name,
        cosmology_numerics=dict(z_max=3.0),
    )
    params = dict(modified_ratio=params_ratio)
    params = complete_params_modified_gravity(kwargs_analysis=kwargs, params=params)

    return name, params


@pytest.fixture(scope="module")
def models(params, numerics, ratio_spec):
    name_modified_ratio, completed_params = ratio_spec
    cosmo_mod = ModifiedGWDistanceFlatLambdaCDM(
        params=params,
        name_modified_ratio=name_modified_ratio,
        params_modified_gravity=completed_params["modified_ratio"],
        numerics=numerics,
    )
    cosmo_astropy = FlatLambdaCDM(H0=params["H0"], Om0=params["Omega_m"])
    return cosmo_mod, cosmo_astropy


# ---------- Tests (run for each ratio_spec) ----------


def test_background_distances_match_astropy(models, z_vals):
    cosmo_mod, cosmo_astropy = models

    d_comoving = cosmo_mod.get_comoving_distance_from_redshift(z_vals)
    d_comoving_astropy = cosmo_astropy.comoving_distance(z_vals).value

    d_luminosity = cosmo_mod.get_luminosity_distance_from_redshift(z_vals)
    d_luminosity_astropy = cosmo_astropy.luminosity_distance(z_vals)

    assert jnp.allclose(d_comoving, d_comoving_astropy, rtol=1e-4, atol=0.0)
    assert jnp.allclose(d_luminosity, d_luminosity_astropy, rtol=1e-4, atol=0.0)


def test_dlum_gw_matches_ratio_times_dlum(models, z_vals):
    cosmo_mod, _ = models

    d_luminosity = cosmo_mod.get_luminosity_distance_from_redshift(z_vals)
    ratio = cosmo_mod.get_ratio_from_redshift(z_vals)
    d_luminosity_gw = cosmo_mod.get_luminosity_distance_gw_from_redshift(z_vals)
    d_luminosity_gw_expected = d_luminosity * ratio

    assert jnp.allclose(d_luminosity_gw, d_luminosity_gw_expected, rtol=5e-6, atol=0.0)


def test_inverse_redshift_recovers_input(models, z_vals):
    cosmo_mod, _ = models

    d_luminosity_gw = cosmo_mod.get_luminosity_distance_gw_from_redshift(z_vals)
    z_inverse = cosmo_mod.get_redshift_from_luminosity_distance_gw(d_luminosity_gw)

    assert jnp.allclose(z_inverse[1:-1], z_vals[1:-1], rtol=0.0, atol=2e-4)


def test_jacobian_matches_finite_difference(models, z_vals):
    cosmo_mod, _ = models

    d_luminosity_gw = cosmo_mod.get_luminosity_distance_gw_from_redshift(z_vals)

    z_vals_centers = (z_vals[1:] + z_vals[:-1]) / 2
    jacobian = cosmo_mod.get_dluminosity_distance_gw_over_dz_from_redshift(
        z_vals_centers
    )

    dz = z_vals[1:] - z_vals[:-1]
    finite_diff = (d_luminosity_gw[1:] - d_luminosity_gw[:-1]) / dz

    assert jnp.allclose(jacobian, finite_diff, rtol=1e-5, atol=0.01)
