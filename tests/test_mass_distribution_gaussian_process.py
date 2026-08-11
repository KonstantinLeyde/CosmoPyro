from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpyro
import pytest
from numpyro.infer.util import log_density

from cosmopyro.distributions.mass_distributions_gaussian_process import (
    construct_prob_nn_whitened_field_2D_logMdelta,
)
from cosmopyro.distributions.source_frame_masses import (
    construct_conditionals_from_prob_logM_delta,
    construct_source_frame_mass_model,
)
from cosmopyro.numpyro_utils.sampling_utils import get_prior_draw
from cosmopyro.utils.utils import get_binning_from_kwargs_analysis

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

NUM_BINS_S = 30
NUM_BINS_DELTA = 30


@jax.jit
def _build_prob_2d(analysis, params):
    return construct_prob_nn_whitened_field_2D_logMdelta(analysis, params)


def _make_analysis_and_params(seed=42):
    kwargs_analysis = dict(
        bins=dict(
            log_mass_total_s=dict(min=1.5, max=4.5, num=NUM_BINS_S),
            minus_log_mass_ratio=dict(min=0.0, max=4.0, num=NUM_BINS_DELTA),
        ),
        distribution_names=dict(),
    )
    analysis = SimpleNamespace()
    analysis.binning = get_binning_from_kwargs_analysis(kwargs_analysis)
    analysis.kwargs_analysis = kwargs_analysis

    key = jax.random.PRNGKey(seed)
    gaussian_F = jax.random.normal(key, shape=(NUM_BINS_S, NUM_BINS_DELTA))

    params = dict(
        source_frame_masses=dict(
            gaussian_F_whitened_spatial=gaussian_F,
            mass_min=5.0,
            mass_max=80.0,
            sigma_low_fractional=0.05,
            sigma_high_fractional=0.05,
            power_spectrum_amplitude=0.045,
            power_spectrum_cutoff=50.0,
            power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio=1.0,
        ),
    )
    return analysis, params


def test_prob_2d_shape():
    analysis, params = _make_analysis_and_params()
    prob = _build_prob_2d(analysis, params)
    assert prob.shape == (NUM_BINS_S, NUM_BINS_DELTA)


def test_prob_2d_non_negative():
    analysis, params = _make_analysis_and_params()
    prob = _build_prob_2d(analysis, params)
    assert jnp.all(prob >= 0.0), f"Found negative probabilities: {prob.min()}"


def test_prob_2d_normalization():
    analysis, params = _make_analysis_and_params()
    prob = _build_prob_2d(analysis, params)

    delta_s = analysis.binning["deltas"]["log_mass_total_s"]
    delta_delta = analysis.binning["deltas"]["minus_log_mass_ratio"]
    integral = jnp.sum(prob * delta_s[:, None] * delta_delta[None, :])

    assert jnp.isclose(integral, 1.0, atol=0.05), (
        f"2D integral was {integral}, expected ~1.0"
    )


def test_prob_2d_determinism():
    analysis, params = _make_analysis_and_params()
    prob1 = _build_prob_2d(analysis, params)
    prob2 = _build_prob_2d(analysis, params)
    assert jnp.array_equal(prob1, prob2)


def test_prob_2d_conditionals_handle_zero_marginal_rows():
    analysis, _ = _make_analysis_and_params()
    prob = jnp.ones((NUM_BINS_S, NUM_BINS_DELTA))
    prob = prob.at[0, :].set(0.0)

    _, prob_delta_given_logM = construct_conditionals_from_prob_logM_delta(
        analysis,
        prob,
    )

    assert jnp.all(jnp.isfinite(prob_delta_given_logM))
    row_integral = jnp.sum(
        prob_delta_given_logM[:, 0] * analysis.binning["deltas"]["minus_log_mass_ratio"]
    )
    assert jnp.isclose(row_integral, 1.0, atol=1e-12)


def test_gp_2d_logMdelta_uses_smooth_log_grid_interpolation_by_default():
    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses": "fourier_gp_2D_logMdelta",
    }

    model_logM, model_delta = construct_source_frame_mass_model(analysis, params)

    assert model_logM.interpolation == "smooth_log"
    assert model_delta.interpolation == "smooth_log"


def test_gp_2d_logMdelta_grid_interpolation_can_use_legacy_linear_density():
    from cosmopyro.utils.analyses import Analysis

    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses": "fourier_gp_2D_logMdelta",
    }
    analysis.kwargs_analysis["bins"]["mass_grid_interpolation"] = "linear_density"

    get_model = Analysis.get_source_frame_mass_model.__get__(analysis)
    model_logM, model_delta = get_model(params)

    assert model_logM.interpolation == "linear_density"
    assert model_delta.interpolation == "linear_density"


# ---------------------------------------------------------------------------
# Gradient-flow tests — catch the silent NaN-gradient failure through the
# zero mode of the power spectrum (sqrt(0) has an undefined derivative;
# without the safe_sqrt fix this poisons gradients of amplitude, cutoff,
# relative_scale with NaN while the forward pass looks fine).
# ---------------------------------------------------------------------------


def _scalar_loss_2d(params, analysis):
    prob = construct_prob_nn_whitened_field_2D_logMdelta(analysis, params)
    return jnp.sum(prob**2)


def _make_source_frame_mass_prior():
    return dict(
        source_frame_masses=dict(
            gaussian_F_whitened_spatial=dict(
                dist_type="Normal",
                loc=0.0,
                scale=1.0,
                shape=[NUM_BINS_S, NUM_BINS_DELTA],
            ),
            mass_min=dict(dist_type="Delta", value=5.0),
            mass_max=dict(dist_type="Delta", value=80.0),
            sigma_low_fractional=dict(dist_type="Delta", value=0.05),
            sigma_high_fractional=dict(dist_type="Delta", value=0.05),
            power_spectrum_amplitude=dict(dist_type="Delta", value=0.045),
            power_spectrum_cutoff=dict(dist_type="Delta", value=50.0),
            power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio=dict(
                dist_type="Delta",
                value=1.0,
            ),
        ),
    )


def _numpyro_gp_2d_model(analysis, prior):
    params = get_prior_draw(prior, use_value_for_Delta=True)
    numpyro.factor("gp_objective", _scalar_loss_2d(params, analysis))


_POWER_SPECTRUM_PARAMS = (
    "power_spectrum_amplitude",
    "power_spectrum_cutoff",
    "power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio",
)


@pytest.mark.parametrize(
    "power_spectrum_name", ["linear_minus_cubic", "quadratic_minus_quartic"]
)
def test_prob_2d_gradients_are_finite(power_spectrum_name):
    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses_power_spectrum": power_spectrum_name,
    }

    grads = jax.grad(_scalar_loss_2d)(params, analysis)["source_frame_masses"]

    for name, g in grads.items():
        g = jnp.asarray(g)
        assert jnp.all(jnp.isfinite(g)), (
            f"[{power_spectrum_name}] gradient w.r.t. {name} is not finite: {g}"
        )


@pytest.mark.parametrize(
    "power_spectrum_name", ["linear_minus_cubic", "quadratic_minus_quartic"]
)
def test_prob_2d_power_spectrum_gradients_nontrivial(power_spectrum_name):
    """The power-spectrum gradients must be non-zero — a zero gradient would
    also hide the original bug (integrator can't learn these parameters)."""
    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses_power_spectrum": power_spectrum_name,
    }

    grads = jax.grad(_scalar_loss_2d)(params, analysis)["source_frame_masses"]
    for name in _POWER_SPECTRUM_PARAMS:
        g = float(grads[name])
        assert jnp.isfinite(g), f"grad[{name}] is not finite: {g}"
        assert abs(g) > 0.0, (
            f"grad[{name}] is exactly zero — inference cannot move this parameter"
        )


# ---------------------------------------------------------------------------
# Whitened-field gradient tests — the whitened field is the dominant source
# of parameters in the GP model (NUM_BINS_S * NUM_BINS_DELTA scalars). If
# any entry silently gets a NaN or zero gradient, HMC / SVI will fail to
# move that pixel and the inferred field will be biased in a way that is
# hard to detect from the chain summary alone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "power_spectrum_name", ["linear_minus_cubic", "quadratic_minus_quartic"]
)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_prob_2d_whitened_field_gradient_per_element(power_spectrum_name, seed):
    """Every pixel of the whitened-field gradient must be finite and at least
    one pixel must be non-zero. Parametrised over seeds to exercise different
    random draws (which change the sign of the whitened field per pixel)."""
    analysis, params = _make_analysis_and_params(seed=seed)
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses_power_spectrum": power_spectrum_name,
    }

    grads = jax.grad(_scalar_loss_2d)(params, analysis)["source_frame_masses"]
    g = jnp.asarray(grads["gaussian_F_whitened_spatial"])

    assert g.shape == (NUM_BINS_S, NUM_BINS_DELTA), f"unexpected grad shape: {g.shape}"
    assert jnp.all(jnp.isfinite(g)), (
        f"[{power_spectrum_name}, seed={seed}] NaN/Inf in whitened-field grad: "
        f"n_bad = {int(jnp.sum(~jnp.isfinite(g)))}"
    )
    assert jnp.any(g != 0.0), (
        f"[{power_spectrum_name}, seed={seed}] whitened-field gradient is "
        f"identically zero — the field would be frozen during inference"
    )


@pytest.mark.parametrize(
    "power_spectrum_name", ["linear_minus_cubic", "quadratic_minus_quartic"]
)
def test_prob_2d_whitened_field_gradient_matches_finite_difference(power_spectrum_name):
    """Verify autograd values for the whitened field match central finite
    differences. Picks a handful of pixels to keep the test fast."""
    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses_power_spectrum": power_spectrum_name,
    }

    grads = jax.grad(_scalar_loss_2d)(params, analysis)["source_frame_masses"]
    g_auto = jnp.asarray(grads["gaussian_F_whitened_spatial"])

    # Corners, centre, off-centre — covers the k=0 mode's influence, edges,
    # and a generic interior pixel.
    pixels = [
        (0, 0),
        (0, NUM_BINS_DELTA - 1),
        (NUM_BINS_S - 1, 0),
        (NUM_BINS_S // 2, NUM_BINS_DELTA // 2),
        (3, 7),
    ]

    eps = 1e-4
    base_w = params["source_frame_masses"]["gaussian_F_whitened_spatial"]
    for i, j in pixels:
        perturb = jnp.zeros_like(base_w).at[i, j].set(eps)

        params_plus = dict(params)
        params_plus["source_frame_masses"] = dict(params["source_frame_masses"])
        params_plus["source_frame_masses"]["gaussian_F_whitened_spatial"] = (
            base_w + perturb
        )

        params_minus = dict(params)
        params_minus["source_frame_masses"] = dict(params["source_frame_masses"])
        params_minus["source_frame_masses"]["gaussian_F_whitened_spatial"] = (
            base_w - perturb
        )

        fd = (
            _scalar_loss_2d(params_plus, analysis)
            - _scalar_loss_2d(params_minus, analysis)
        ) / (2 * eps)
        auto = float(g_auto[i, j])

        assert jnp.isclose(fd, auto, rtol=1e-4, atol=1e-6), (
            f"[{power_spectrum_name}] pixel ({i},{j}): autograd={auto}, "
            f"finite-diff={float(fd)}"
        )


@pytest.mark.parametrize(
    "power_spectrum_name", ["linear_minus_cubic", "quadratic_minus_quartic"]
)
def test_prob_2d_whitened_field_gradient_coverage(power_spectrum_name):
    """At least 90% of whitened-field pixels must have non-trivial (>1e-10)
    gradients. A few near-zero entries are acceptable (e.g. the window may
    effectively mask some pixels), but wholesale zeroing indicates a
    broken chain rule."""
    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses_power_spectrum": power_spectrum_name,
    }

    grads = jax.grad(_scalar_loss_2d)(params, analysis)["source_frame_masses"]
    g = jnp.asarray(grads["gaussian_F_whitened_spatial"])

    n_total = g.size
    n_nonzero = int(jnp.sum(jnp.abs(g) > 1e-10))
    frac = n_nonzero / n_total

    assert frac >= 0.9, (
        f"[{power_spectrum_name}] only {n_nonzero}/{n_total} "
        f"({frac:.1%}) of whitened-field pixels have gradient > 1e-10"
    )


@pytest.mark.parametrize(
    "power_spectrum_name", ["linear_minus_cubic", "quadratic_minus_quartic"]
)
def test_numpyro_log_density_whitened_field_gradient_is_finite(power_spectrum_name):
    """Exercise the actual NumPyro latent site used in inference.

    This guards against a silent break where the GP field remains visible to
    raw JAX autodiff but stops contributing gradients once wrapped in a
    NumPyro model.
    """
    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses_power_spectrum": power_spectrum_name,
    }
    prior = _make_source_frame_mass_prior()

    position = {
        "gaussian_F_whitened_spatial_white": params["source_frame_masses"][
            "gaussian_F_whitened_spatial"
        ]
    }

    grad_fn = jax.grad(
        lambda pos: log_density(_numpyro_gp_2d_model, (analysis, prior), {}, pos)[0]
    )
    g = grad_fn(position)["gaussian_F_whitened_spatial_white"]

    assert g.shape == (NUM_BINS_S, NUM_BINS_DELTA)
    assert jnp.all(jnp.isfinite(g)), (
        f"[{power_spectrum_name}] NaN/Inf in NumPyro white-site gradient: "
        f"n_bad = {int(jnp.sum(~jnp.isfinite(g)))}"
    )
    assert jnp.any(g != 0.0), (
        f"[{power_spectrum_name}] NumPyro white-site gradient is identically zero"
    )


@pytest.mark.parametrize(
    "power_spectrum_name", ["linear_minus_cubic", "quadratic_minus_quartic"]
)
def test_numpyro_log_density_whitened_field_gradient_matches_direct_gradient(
    power_spectrum_name,
):
    """For loc=0, scale=1 priors, NumPyro's white-site gradient should equal
    the direct GP gradient minus the Normal prior score (-white)."""
    analysis, params = _make_analysis_and_params()
    analysis.kwargs_analysis["distribution_names"] = {
        "source_frame_masses_power_spectrum": power_spectrum_name,
    }
    prior = _make_source_frame_mass_prior()

    white = params["source_frame_masses"]["gaussian_F_whitened_spatial"]
    position = {"gaussian_F_whitened_spatial_white": white}

    grad_numpyro = jax.grad(
        lambda pos: log_density(_numpyro_gp_2d_model, (analysis, prior), {}, pos)[0]
    )(position)["gaussian_F_whitened_spatial_white"]

    grad_direct = jax.grad(_scalar_loss_2d)(params, analysis)["source_frame_masses"][
        "gaussian_F_whitened_spatial"
    ]
    expected = grad_direct - white

    assert jnp.allclose(grad_numpyro, expected, rtol=1e-5, atol=1e-6), (
        f"[{power_spectrum_name}] max |NumPyro - expected| = "
        f"{float(jnp.max(jnp.abs(grad_numpyro - expected)))}"
    )
