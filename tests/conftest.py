from types import SimpleNamespace

import jax
import pytest

# Enable x64 as per your script
jax.config.update("jax_enable_x64", True)

# --- Imports from your library ---
from cosmopyro.cosmology.cosmology import FlatLambdaCDM
from cosmopyro.utils.utils import get_binning_from_kwargs_analysis


@pytest.fixture
def analysis_setup():
    """Fixture for the analysis object."""
    analysis = SimpleNamespace()
    analysis.binning = get_binning_from_kwargs_analysis()
    return analysis


@pytest.fixture
def default_params(analysis_setup):
    """Fixture for default parameters."""
    fraction_power_law = 0.1  # Set non-zero for testing interesting cases

    params = dict(
        mass_1_s=dict(
            mass_min=7.0,
            mass_max=50.0,
            alpha_0=-3.0,
            fraction_power_law=fraction_power_law,
            power_spectrum_amplitude=100000,
            power_spectrum_cutoff=0.01,
            sigma_low_fractional=0.02,
            sigma_high_fractional=0.02,
        )
    )

    # Generate a fixed random field for consistent testing
    shape = analysis_setup.binning["deltas"]["mass_1_s"].shape[0]
    key = jax.random.PRNGKey(123)
    gaussian_F = jax.random.normal(key, shape=(shape,))

    params["mass_1_s"]["gaussian_F_whitened_spatial"] = gaussian_F
    return params


@pytest.fixture
def cosmology_fixture():
    params = dict(H0=67.0, Omega_m=0.3)
    return FlatLambdaCDM(
        params=params, numerics={"z_min": 1e-07, "z_max": 3.0, "z_steps": 200000}
    )
