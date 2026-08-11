# tests/test_cosmology.py
import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from cosmopyro.cosmology import flat_lambdacdm as cosmopyro_cosmology
from cosmopyro.cosmology.flat_lambdacdm import C_KM_PER_SEC

# --- Optional dependency: Astropy ---
try:
    from astropy.cosmology import FlatLambdaCDM

    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False


# -------------------------
# Core distance sanity tests
# -------------------------
H0_REF = 67.0
Z_REF = 1.0
LUM_DIST_AT_Z1 = 6903.8  # expected value
Z_SMALL = 1e-5
COMOVING_SMALL = C_KM_PER_SEC / H0_REF * Z_SMALL


class TestCoreDistances:
    def test_luminosity_distance_scalar_at_z1(self, cosmology_fixture):
        cosmo = cosmology_fixture
        d_l = cosmo.get_luminosity_distance_from_redshift(Z_REF)
        assert pytest.approx(d_l, rel=0.001) == LUM_DIST_AT_Z1, (
            "Luminosity distance at z=1 is incorrect"
        )

    def test_small_z_limit_luminosity_equals_comoving(self, cosmology_fixture):
        cosmo = cosmology_fixture
        d_l_small = cosmo.get_luminosity_distance_from_redshift(Z_SMALL)
        assert pytest.approx(d_l_small, rel=0.001) == COMOVING_SMALL, (
            "Small-z luminosity distance (≈ comoving) mismatch"
        )

    def test_roundtrip_redshift_from_luminosity(self, cosmology_fixture):
        cosmo = cosmology_fixture
        d_l = cosmo.get_luminosity_distance_from_redshift(Z_REF)
        z_back = cosmo.get_redshift_from_luminosity_distance(d_l)
        assert pytest.approx(z_back, rel=0.001) == Z_REF, (
            "Redshift from luminosity distance is incorrect"
        )

    @pytest.mark.parametrize("z_vals", [jnp.array([1e-5, 0.1, 0.5, 1.0, 2.0])])
    def test_vectorized_interface_luminosity(self, cosmology_fixture, z_vals):
        cosmo = cosmology_fixture
        out = cosmo.get_luminosity_distance_from_redshift(z_vals)
        assert out.shape == z_vals.shape


# -----------------------------------
# Cross-validation against Astropy ΛCDM
# -----------------------------------
PARAMS = dict(H0=70.0, Omega_m=0.3)
NUMERICS = dict(z_min=1e-5, z_max=3.0, z_steps=200_000)
RTOL = 1e-4


# --- small helper for centers and diffs on jnp arrays ---
def _centers_and_diffs(z: jnp.ndarray):
    z = jnp.asarray(z)
    diffs = z[1:] - z[:-1]
    centers = 0.5 * (z[1:] + z[:-1])
    return centers, diffs


@pytest.fixture(scope="module")
def z_grid():
    # Dense but CI-friendly; tweak if runtime is high
    return jnp.linspace(1e-5, 2.0, 50_000)


@pytest.fixture(scope="module")
def cosmo_models():
    cosmo = cosmopyro_cosmology.FlatLambdaCDM(params=PARAMS, numerics=NUMERICS)
    if not HAS_ASTROPY:
        pytest.skip("Astropy not available for reference comparison")
    cosmo_astropy = FlatLambdaCDM(H0=PARAMS["H0"], Om0=PARAMS["Omega_m"])
    return cosmo, cosmo_astropy


@pytest.mark.skipif(not HAS_ASTROPY, reason="Astropy not available")
class TestAgainstAstropy:
    def test_comoving_distance_matches_astropy(self, cosmo_models, z_grid):
        cosmo, cosmo_astropy = cosmo_models
        d_comoving = cosmo.get_comoving_distance_from_redshift(z_grid)
        d_comoving_ref = jnp.asarray(cosmo_astropy.comoving_distance(z_grid).value)
        assert jnp.allclose(d_comoving, d_comoving_ref, rtol=RTOL), (
            "Comoving distance deviates from Astropy FlatLambdaCDM reference"
        )

    def test_luminosity_distance_matches_astropy(self, cosmo_models, z_grid):
        cosmo, cosmo_astropy = cosmo_models
        d_lum = cosmo.get_luminosity_distance_from_redshift(z_grid)
        d_lum_ref = jnp.asarray(cosmo_astropy.luminosity_distance(z_grid))
        assert jnp.allclose(d_lum, d_lum_ref, rtol=RTOL), (
            "Luminosity distance deviates from Astropy FlatLambdaCDM reference"
        )

    def test_dcomoving_dz_matches_astropy_finite_difference(self, cosmo_models, z_grid):
        cosmo, cosmo_astropy = cosmo_models
        d_comoving_ref = jnp.asarray(cosmo_astropy.comoving_distance(z_grid).value)
        z_centers, z_diffs = _centers_and_diffs(z_grid)
        dchi_dz_fd = (d_comoving_ref[1:] - d_comoving_ref[:-1]) / z_diffs
        dchi_dz_model = cosmo.get_dcomoving_distance_over_dz_from_redshift(z_centers)
        assert jnp.allclose(dchi_dz_model, dchi_dz_fd, rtol=1e-5, atol=1.0), (
            "d(comoving)/dz (model) does not match Astropy finite-difference"
        )


# --------------------------------------------------
# Jacobians vs JAX autodiff on jnp arrays (no Astropy)
# --------------------------------------------------
class TestJacobiansJAX:
    def test_dluminosity_dz_matches_jax_grad(self):
        cosmo = cosmopyro_cosmology.FlatLambdaCDM(params=PARAMS, numerics=NUMERICS)
        z_vals = jnp.linspace(1e-5, 2.0, 100_000)
        z_centers, _ = _centers_and_diffs(z_vals)
        dDL_dz_model = cosmo.get_dluminosity_distance_over_dz_from_redshift(z_centers)
        dDL_dz_grad = jax.vmap(jax.grad(cosmo.get_luminosity_distance_from_redshift))(
            z_centers
        )
        assert jnp.allclose(dDL_dz_model, dDL_dz_grad, rtol=1e-5, atol=0.0), (
            "d(luminosity distance)/dz (model) does not match JAX gradient"
        )

    def test_gradient_through_H0(self):
        """Verify JAX can differentiate luminosity distance w.r.t. H0."""

        def dL_at_z01(H0):
            c = cosmopyro_cosmology.FlatLambdaCDM(params={"H0": H0, "Omega_m": 0.3})
            return c.get_luminosity_distance_from_redshift(0.1)

        grad_dL_dH0 = jax.grad(dL_at_z01)(70.0)
        # dL ∝ 1/H0 so ddL/dH0 ≈ -dL/H0
        expected = -dL_at_z01(70.0) / 70.0
        assert pytest.approx(float(grad_dL_dH0), rel=1e-4) == float(expected)
