from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmopyro.distributions.grid_distributions import (
    InterpolatedConditional1D,
    normalize_cond_interpolated_1d,
)
from cosmopyro.distributions.mass_distribution_parametrized import (
    LVK_SMOOTHING_FLOOR,
    construct_prob_nn_multipeak_1D,
    construct_prob_nn_power_law_peak_1D,
    construct_running_power_law_prob_mass_ratio_nn,
    log_low_mass_smoothing,
    log_smoothing_low_mass_lvk,
)
from cosmopyro.distributions.source_frame_masses import (
    construct_source_frame_mass_model,
    rename_legacy_distribution_names,
    split_smoothing,
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


@pytest.mark.parametrize("smoothing", ["gaussian", "lvk"])
def test_power_law_peak_integration(analysis_coarse, params_all, smoothing):
    cond = construct_prob_nn_power_law_peak_1D(
        analysis_coarse, params_all["power_law_peak"], smoothing=smoothing
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


@pytest.mark.parametrize("smoothing", ["gaussian", "lvk"])
def test_multipeak_integration(analysis_coarse, params_all, smoothing):
    cond = construct_prob_nn_multipeak_1D(
        analysis_coarse, params_all["power_law_peak2"], smoothing=smoothing
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


@pytest.mark.parametrize("smoothing", ["gaussian", "lvk"])
def test_mass_ratio_conditional_integration(analysis_coarse, params_all, smoothing):
    cond = construct_running_power_law_prob_mass_ratio_nn(
        analysis_coarse, params_all, smoothing=smoothing
    )
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


@pytest.mark.parametrize("smoothing", ["gaussian", "lvk"])
def test_mass_ratio_conditional_has_nonzero_gradient_in_conditioning_variable(
    analysis_coarse, params_all, smoothing
):
    cond = construct_running_power_law_prob_mass_ratio_nn(
        analysis_coarse, params_all, smoothing=smoothing
    )
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


@pytest.mark.parametrize("smoothing", ["gaussian", "lvk"])
def test_param_selection_logic(analysis_coarse, params_all, smoothing):
    cond_1 = construct_prob_nn_power_law_peak_1D(
        analysis_coarse, params_all["power_law_peak"], smoothing=smoothing
    )
    assert cond_1 is not None

    cond_2 = construct_prob_nn_multipeak_1D(
        analysis_coarse, params_all["power_law_peak2"], smoothing=smoothing
    )
    assert cond_2 is not None


@pytest.mark.parametrize("smoothing", ["gaussian", "lvk"])
def test_smoothing_boundary_fine_check(analysis_coarse, params_all, smoothing):
    cond = construct_prob_nn_power_law_peak_1D(
        analysis_coarse, params_all["power_law_peak"], smoothing=smoothing
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


# -------------------------------------------------------------------------
# Low-mass smoothing: LVK turn-on and model names
# -------------------------------------------------------------------------


def _lvk_smoothing_reference(m, mmin, delta_m):
    """Plain NumPy LVK turn-on (Talbot & Thrane 2018), without floor."""
    m = np.asarray(m, dtype=float)
    out = np.where(m >= mmin + delta_m, 1.0, 0.0)
    inside = (m > mmin) & (m < mmin + delta_m)
    mp = m[inside] - mmin
    with np.errstate(over="ignore"):  # exp overflows to inf right at mmin, giving S = 0
        out[inside] = 1.0 / (1.0 + np.exp(delta_m / mp + delta_m / (mp - delta_m)))
    return out


def test_lvk_smoothing_matches_reference_and_floor():
    mmin, delta_m = 5.0, 3.0
    m = np.linspace(2.0, 10.0, 4001)
    s = np.exp(np.asarray(log_smoothing_low_mass_lvk(jnp.asarray(m), mmin, delta_m)))
    s_ref = _lvk_smoothing_reference(m, mmin, delta_m)

    np.testing.assert_allclose(s, np.maximum(s_ref, LVK_SMOOTHING_FLOOR), rtol=1e-12)
    assert np.all(s[m <= mmin] == pytest.approx(LVK_SMOOTHING_FLOOR, rel=1e-12))
    assert np.all(s[m >= mmin + delta_m] == 1.0)
    half = float(jnp.exp(log_smoothing_low_mass_lvk(mmin + delta_m / 2, mmin, delta_m)))
    assert half == pytest.approx(0.5, rel=1e-12)


def test_lvk_smoothing_gradients_finite_everywhere():
    mmin, delta_m = 5.0, 3.0
    # include the exact edges of the turn-on and points just inside / outside
    eps = 1e-12
    m = jnp.concatenate(
        [
            jnp.linspace(2.0, 10.0, 2001),
            jnp.array([mmin, mmin + eps, mmin + delta_m - eps, mmin + delta_m]),
        ]
    )

    def f(mmin, delta_m, m):
        return jnp.sum(log_smoothing_low_mass_lvk(m, mmin, delta_m))

    g_mmin, g_delta_m, g_m = jax.grad(f, argnums=(0, 1, 2))(mmin, delta_m, m)
    assert jnp.isfinite(g_mmin) and jnp.isfinite(g_delta_m)
    assert jnp.all(jnp.isfinite(g_m))
    assert g_mmin != 0.0 and g_delta_m != 0.0


def test_unknown_smoothing_raises():
    with pytest.raises(ValueError, match="Unknown low-mass smoothing"):
        log_low_mass_smoothing("polynomial", jnp.array([6.0]), 5.0, 2.0)


def test_split_smoothing_and_legacy_names():
    assert split_smoothing("power_law_peak2_partial_windowed_smoothing_lvk") == (
        "power_law_peak2_partial_windowed",
        "lvk",
    )
    assert split_smoothing("fourier_gp_1D") == ("fourier_gp_1D", None)

    kwargs = dict(
        distribution_names=dict(
            mass_1_s="power_law_peak2_partial_windowed",
            mass_ratio="mass_ratio_running_power_law_in_log",
            redshift="MadauDickinson",
        )
    )
    with pytest.warns(UserWarning, match="Legacy distribution name"):
        rename_legacy_distribution_names(kwargs)
    assert kwargs["distribution_names"] == dict(
        mass_1_s="power_law_peak2_partial_windowed_smoothing_gaussian",
        mass_ratio="mass_ratio_running_power_law_in_log_smoothing_gaussian",
        redshift="MadauDickinson",
    )


def _analysis_with_names(analysis_coarse, mass_1_s, mass_ratio):
    analysis = SimpleNamespace(**vars(analysis_coarse))
    analysis.kwargs_analysis = dict(
        distribution_names=dict(mass_1_s=mass_1_s, mass_ratio=mass_ratio)
    )
    analysis.get_distribution_name = lambda key: analysis.kwargs_analysis[
        "distribution_names"
    ][key]
    return analysis


def _params_power_law_peak2(params_all, **mass_1_s):
    return dict(
        mass_1_s={**params_all["power_law_peak2"]["mass_1_s"], **mass_1_s},
        mass_ratio=params_all["mass_ratio"],
    )


def test_mixed_smoothing_raises(analysis_coarse, params_all):
    analysis = _analysis_with_names(
        analysis_coarse,
        "power_law_peak2_smoothing_lvk",
        "mass_ratio_running_power_law_in_log_smoothing_gaussian",
    )
    with pytest.raises(ValueError, match="Low-mass smoothing differs"):
        construct_source_frame_mass_model(
            analysis, _params_power_law_peak2(params_all)
        )


@pytest.mark.parametrize("partial", ["", "_partial_windowed"])
def test_lvk_mass_model_log_prob_gradients_finite(analysis_coarse, params_all, partial):
    """End to end: log p(m1) + log p(q | m1) and its gradient in mmin / delta_m stay finite,
    also for samples below mmin and with m2 below mmin."""
    analysis = _analysis_with_names(
        analysis_coarse,
        f"power_law_peak2{partial}_smoothing_lvk",
        "mass_ratio_running_power_law_in_log_smoothing_lvk",
    )
    m1 = jnp.linspace(6.0, 95.0, 200)
    q = jnp.linspace(0.05, 0.99, 200)

    def total_log_prob(mmin, delta_m):
        params = _params_power_law_peak2(params_all, mmin=mmin, delta_m=delta_m)
        model_m1, model_q = construct_source_frame_mass_model(analysis, params)
        return jnp.sum(
            model_m1.log_prob(x_vals={"mass_1_s": m1})
            + model_q.log_prob(x_vals={"mass_ratio": q}, y_vals={"mass_1_s": m1})
        )

    value, grads = jax.value_and_grad(total_log_prob, argnums=(0, 1))(10.0, 2.0)
    assert jnp.isfinite(value)
    assert all(jnp.isfinite(g) for g in grads)
