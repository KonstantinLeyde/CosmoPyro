from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from cosmopyro.distributions.grid_distributions import (
    InterpolatedConditional1D,
    normalize_cond_interpolated_1d,
)
from cosmopyro.distributions.mass_distribution_parametrized import (
    construct_prob_nn_multipeak_1D,
    construct_prob_nn_power_law_peak_1D,
    construct_running_power_law_prob_mass_ratio_nn,
)

jax.config.update("jax_enable_x64", True)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def analysis_coarse():
    m_bins = jnp.linspace(5.0, 100.0, 50)
    q_bins = jnp.linspace(0.01, 1.0, 20)

    analysis = SimpleNamespace()
    analysis.binning = {
        "boundaries": {"mass_1_s": m_bins, "mass_ratio": q_bins},
        "centers": {
            "mass_1_s": 0.5 * (m_bins[1:] + m_bins[:-1]),
            "mass_ratio": 0.5 * (q_bins[1:] + q_bins[:-1]),
        },
    }
    return analysis


@pytest.fixture
def params_all():
    return dict(
        power_law_peak=dict(
            mass_1_s=dict(
                alpha=3.0,
                mmin=10.0,
                mmax=80.0,
                mu_g=35.0,
                sigma_g=3.0,
                lambda_peak=0.1,
                delta_m=2.0,
            ),
        ),
        power_law_peak2=dict(
            mass_1_s=dict(
                alpha=2.5,
                mmin=10.0,
                mmax=80.0,
                mu_g_low=20.0,
                sigma_g_low=2.0,
                lambda_g_low=0.5,
                mu_g_high=45.0,
                sigma_g_high=4.0,
                lambda_g=0.2,
                delta_m=2.0,
            ),
        ),
        mass_ratio=dict(
            beta_0=1.0,
            beta_1=0.5,
            sigma_mass_cutoff_mass_2=1.5,
            mass_ratio_running_zero_point=10.0,
        ),
        mass_1_s=dict(mass_min=5.0),
    )


# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------


def integrate_model_on_fine_grid(
    model, param_name, min_val, max_val, condition_dict=None
):
    fine_x = jnp.linspace(min_val, max_val, 50000)
    x_vals = {param_name: fine_x}
    y_vals = condition_dict
    log_probs = model.log_prob(x_vals=x_vals, y_vals=y_vals)
    probs = jnp.exp(log_probs)
    return jnp.trapezoid(probs, fine_x)


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_normalize_cond_interpolated_1d_preserves_tiny_positive_mass():
    edges = jnp.array([0.0, 1.0])
    cond = jnp.array([1e-20])

    cond_norm = normalize_cond_interpolated_1d(edges, cond)
    integral = cond_norm[0] * (edges[1] - edges[0])

    assert jnp.isclose(integral, 1.0, rtol=1e-12, atol=1e-12)


def test_power_law_peak_integration(analysis_coarse, params_all):
    cond = construct_prob_nn_power_law_peak_1D(
        analysis_coarse, params_all["power_law_peak"]
    )
    cond = normalize_cond_interpolated_1d(
        x_edges=analysis_coarse.binning["boundaries"]["mass_1_s"],
        cond=cond,
    )
    model = InterpolatedConditional1D(
        x_bins={"mass_1_s": analysis_coarse.binning["boundaries"]["mass_1_s"]},
        y_bins=None,
        cond=cond,
    )
    m_min = params_all["power_law_peak"]["mass_1_s"]["mmin"]
    m_max = params_all["power_law_peak"]["mass_1_s"]["mmax"]
    area = integrate_model_on_fine_grid(model, "mass_1_s", m_min, m_max)
    assert jnp.isclose(area, 1.0, atol=0.05), f"Single Peak integration failed: {area}"


def test_multipeak_integration(analysis_coarse, params_all):
    cond = construct_prob_nn_multipeak_1D(
        analysis_coarse, params_all["power_law_peak2"]
    )
    cond = normalize_cond_interpolated_1d(
        x_edges=analysis_coarse.binning["boundaries"]["mass_1_s"],
        cond=cond,
    )
    model = InterpolatedConditional1D(
        x_bins={"mass_1_s": analysis_coarse.binning["boundaries"]["mass_1_s"]},
        y_bins=None,
        cond=cond,
    )
    m_min = params_all["power_law_peak2"]["mass_1_s"]["mmin"]
    m_max = params_all["power_law_peak2"]["mass_1_s"]["mmax"]
    area = integrate_model_on_fine_grid(model, "mass_1_s", m_min, m_max)
    assert jnp.isclose(area, 1.0, atol=0.05), f"Multipeak integration failed: {area}"


def test_mass_ratio_conditional_integration(analysis_coarse, params_all):
    cond = construct_running_power_law_prob_mass_ratio_nn(analysis_coarse, params_all)
    model = InterpolatedConditional1D(
        x_bins={"mass_ratio": analysis_coarse.binning["boundaries"]["mass_ratio"]},
        y_bins={"mass_1_s": analysis_coarse.binning["boundaries"]["mass_1_s"]},
        cond=cond,
        continuous_y_names=["mass_1_s"],
    )
    test_masses = [20.0, 40.0, 60.0]
    for m_val in test_masses:
        area = integrate_model_on_fine_grid(
            model, "mass_ratio", 0.001, 1.0, condition_dict={"mass_1_s": m_val}
        )
        assert jnp.isclose(area, 1.0, rtol=1e-4, atol=1e-4), (
            f"Mass ratio integration failed at m1={m_val}. Area: {area}"
        )


def test_mass_ratio_conditional_has_nonzero_gradient_in_conditioning_variable(
    analysis_coarse, params_all
):
    cond = construct_running_power_law_prob_mass_ratio_nn(analysis_coarse, params_all)
    model = InterpolatedConditional1D(
        x_bins={"mass_ratio": analysis_coarse.binning["boundaries"]["mass_ratio"]},
        y_bins={"mass_1_s": analysis_coarse.binning["boundaries"]["mass_1_s"]},
        cond=cond,
        continuous_y_names=["mass_1_s"],
    )

    def f(mass_1_s):
        return model.log_prob(
            x_vals={"mass_ratio": jnp.array(0.6)},
            y_vals={"mass_1_s": mass_1_s},
        )

    grads = jax.vmap(jax.grad(f))(jnp.linspace(20.0, 60.0, 11))
    assert jnp.any(jnp.abs(grads) > 1e-8), (
        f"Expected non-zero conditioning gradient, got {grads}"
    )


def test_param_selection_logic(analysis_coarse, params_all):
    cond_1 = construct_prob_nn_power_law_peak_1D(
        analysis_coarse, params_all["power_law_peak"]
    )
    assert cond_1 is not None

    cond_2 = construct_prob_nn_multipeak_1D(
        analysis_coarse, params_all["power_law_peak2"]
    )
    assert cond_2 is not None


def test_smoothing_boundary_fine_check(analysis_coarse, params_all):
    cond = construct_prob_nn_power_law_peak_1D(
        analysis_coarse, params_all["power_law_peak"]
    )
    model = InterpolatedConditional1D(
        x_bins={"mass_1_s": analysis_coarse.binning["boundaries"]["mass_1_s"]},
        y_bins=None,
        cond=cond,
    )
    m_min = params_all["power_law_peak"]["mass_1_s"]["mmin"]

    prob_below = jnp.exp(model.log_prob(x_vals={"mass_1_s": jnp.array([m_min - 0.5])}))
    prob_above = jnp.exp(model.log_prob(x_vals={"mass_1_s": jnp.array([m_min + 0.5])}))

    # Smooth turn-on leaks a little just below mmin on this coarse grid, so
    # require strong relative suppression rather than a hard absolute floor.
    assert prob_above > 0.0, "Probability just above mmin should be positive."
    assert prob_below < 0.01 * prob_above, (
        f"Density below mmin not suppressed: {prob_below}"
    )
