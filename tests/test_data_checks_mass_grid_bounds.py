from types import SimpleNamespace

import jax.numpy as jnp

from cosmopyro.utils.utils import (
    check_cosmology_has_high_enough_redshift_coverage,
    check_source_frame_masses_within_grid_bounds,
)


class FakeCosmology:
    def __init__(self, H0):
        self.H0 = H0

    def get_redshift_from_luminosity_distance(self, luminosity_distance):
        return luminosity_distance / self.H0

    def get_luminosity_distance_from_redshift(self, redshift):
        return redshift * self.H0


class FakeAnalysis:
    kwargs_analysis = {
        "cosmology_numerics": {"z_max": 2.0},
        "kwargs_priors": {
            "cosmology": {
                "h": {"dist_type": "Uniform", "min": 0.5, "max": 1.0},
                "Omega_m": {"dist_type": "Uniform", "min": 0.2, "max": 0.4},
            },
        },
    }
    binning = {
        "boundaries": {
            "mass_1_s": jnp.array([10.0, 20.0, 30.0]),
            "mass_ratio": jnp.array([0.1, 0.5, 1.0]),
        },
    }

    def get_cosmological_model(self, parameters):
        return FakeCosmology(parameters["cosmology"]["H0"])


class FakeLogMassAnalysis(FakeAnalysis):
    binning = {
        "boundaries": {
            "log_mass_total_s": jnp.linspace(1.0, 6.0, 421),
            "minus_log_mass_ratio": jnp.linspace(0.0, 3.0, 421),
        },
    }


def test_source_frame_mass_grid_check_uses_cosmology_prior_corners():
    data = SimpleNamespace(
        samples=SimpleNamespace(
            mass_1_d=jnp.array([70.0]),
            mass_ratio=jnp.array([0.5]),
            luminosity_distance=jnp.array([100.0]),
        ),
        injections=SimpleNamespace(
            mass_1_d=jnp.array([18.0]),
            mass_ratio=jnp.array([0.5]),
            luminosity_distance=jnp.array([10.0]),
        ),
    )

    errors = check_source_frame_masses_within_grid_bounds(
        FakeAnalysis(),
        {"data": data},
    )

    assert any("PE samples.mass_1_s" in error for error in errors)
    assert any("H0=100" in error for error in errors)


def test_source_frame_mass_grid_check_flags_mass_ratio_bounds():
    data = SimpleNamespace(
        samples=SimpleNamespace(
            mass_1_d=jnp.array([20.0]),
            mass_ratio=jnp.array([1.2]),
            luminosity_distance=jnp.array([10.0]),
        ),
    )

    errors = check_source_frame_masses_within_grid_bounds(
        FakeAnalysis(),
        {"data": data},
    )

    assert any("samples.mass_ratio" in error for error in errors)


def test_source_frame_mass_grid_check_supports_log_mass_total_grid():
    data = SimpleNamespace(
        samples=SimpleNamespace(
            mass_1_d=jnp.array([20.0]),
            mass_ratio=jnp.array([0.5]),
            luminosity_distance=jnp.array([10.0]),
        ),
        injections=SimpleNamespace(
            mass_1_d=jnp.array([20.0]),
            mass_ratio=jnp.array([0.5]),
            luminosity_distance=jnp.array([10.0]),
        ),
    )

    errors = check_source_frame_masses_within_grid_bounds(
        FakeLogMassAnalysis(),
        {"data": data},
    )

    assert errors == []


def test_source_frame_mass_grid_check_flags_log_mass_total_bounds():
    data = SimpleNamespace(
        samples=SimpleNamespace(
            mass_1_d=jnp.array([800.0]),
            mass_ratio=jnp.array([0.5]),
            luminosity_distance=jnp.array([10.0]),
        ),
    )

    errors = check_source_frame_masses_within_grid_bounds(
        FakeLogMassAnalysis(),
        {"data": data},
    )

    assert any("samples.log_mass_total_s" in error for error in errors)


def test_redshift_coverage_check_uses_cosmology_prior_corners():
    data = SimpleNamespace(
        samples=SimpleNamespace(luminosity_distance=jnp.array([150.0])),
        injections=SimpleNamespace(luminosity_distance=jnp.array([10.0])),
    )

    errors = check_cosmology_has_high_enough_redshift_coverage(
        FakeAnalysis(),
        {"data": data},
    )

    assert any("H0=50" in error for error in errors)
    assert not any("H0=100" in error for error in errors)
