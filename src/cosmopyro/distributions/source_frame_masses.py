import jax.numpy as jnp

from .grid_distributions import (
    InterpolatedConditional1D,
    normalize_cond_interpolated_1d,
)
from .mass_distribution_parametrized import (
    construct_prob_nn_multipeak_1D,
    construct_prob_nn_power_law_peak_1D,
    construct_prob_nn_truncated_gaussian_mass_ratio,
    construct_running_power_law_prob_mass_ratio_nn,
    get_log_window_mass_s,
)
from .mass_distributions_gaussian_process import (
    construct_log_prob_nn_whitened_field_1D,
    construct_prob_nn_whitened_field_2D_logMdelta,
    construct_prob_nn_whitened_field_2D_m1sq,
)

__all__ = [
    "construct_conditionals_from_prob_logM_delta",
    "construct_conditionals_from_prob_mass_1_s_mass_ratio",
    "construct_mass_1_s_prob_nn",
    "construct_mass_ratio_prob_nn",
    "construct_source_frame_mass_model",
]


def _safe_conditional_from_joint(joint_prob, marginal_prob, x_widths):
    """Return p(x | y) from p(y, x), avoiding NaNs for zero-marginal rows.

    The guard is ``> sqrt(tiny)``, not ``> 0``. Reverse mode differentiates the
    division as ``-joint / marginal**2``; below sqrt of the smallest normal
    (~1.5e-154 in float64) that square underflows to zero, so the cotangent is
    inf and every parameter feeding the mass model gets a NaN gradient -- while
    the forward pass still looks perfectly finite. Rows that far down carry no
    probability mass, so they belong in the uniform fallback anyway.
    """
    floor = jnp.sqrt(jnp.finfo(marginal_prob.dtype).tiny)
    marginal_positive = marginal_prob > floor
    marginal_safe = jnp.where(marginal_positive, marginal_prob, 1.0)
    conditional = joint_prob / marginal_safe[:, None]

    uniform_density = jnp.ones_like(joint_prob) / jnp.sum(x_widths)
    return jnp.where(marginal_positive[:, None], conditional, uniform_density)


def construct_source_frame_mass_model(analysis, params, interpolation="smooth_log"):

    distribution_names = analysis.kwargs_analysis["distribution_names"]
    source_frame_masses_name = distribution_names.get("source_frame_masses", None)

    if source_frame_masses_name is not None:
        # Joint mass model in transformed coordinates
        return _construct_joint_mass_model(
            analysis, params, source_frame_masses_name, interpolation=interpolation
        )

    # Factorized mass model: separate mass_1_s and mass_ratio distributions
    mass_1_s_name = analysis.get_distribution_name("mass_1_s")
    mass_ratio_name = analysis.get_distribution_name("mass_ratio")

    if mass_1_s_name in [
        "power_law_peak",
        "power_law_peak2",
        "power_law_peak2_partial_windowed",
        "fourier_gp_1D",
    ] and mass_ratio_name in [
        "mass_ratio_running_power_law_in_log",
        "mass_ratio_truncated_gaussian",
    ]:
        cond = construct_mass_1_s_prob_nn(analysis, params)
        prob_m1s = normalize_cond_interpolated_1d(
            x_edges=analysis.binning["boundaries"]["mass_1_s"],
            cond=cond,
        )

        cond = construct_mass_ratio_prob_nn(analysis, params)
        prob_mass_ratio_given_mass_1_s_qm1s = normalize_cond_interpolated_1d(
            x_edges=analysis.binning["boundaries"]["mass_ratio"],
            cond=cond,
        )

    else:
        raise NotImplementedError(
            f"Mass model combination mass_1_s={mass_1_s_name} and mass_ratio={mass_ratio_name} not implemented."
        )

    model_mass_1_s = InterpolatedConditional1D(
        x_bins={"mass_1_s": analysis.binning["boundaries"]["mass_1_s"]},
        y_bins=None,
        cond=prob_m1s,
        interpolation=interpolation,
    )
    model_mass_ratio = InterpolatedConditional1D(
        x_bins={"mass_ratio": analysis.binning["boundaries"]["mass_ratio"]},
        y_bins={"mass_1_s": analysis.binning["boundaries"]["mass_1_s"]},
        cond=prob_mass_ratio_given_mass_1_s_qm1s,
        continuous_y_names=["mass_1_s"],
        interpolation=interpolation,
    )

    return model_mass_1_s, model_mass_ratio


def _construct_joint_mass_model(
    analysis, params, source_frame_masses_name, interpolation="smooth_log"
):

    if source_frame_masses_name in ["fourier_gp_2D_logMdelta"]:
        prob_logM_delta_nn = construct_prob_nn_whitened_field_2D_logMdelta(
            analysis, params
        )

        prob_logM, prob_delta_given_logM = construct_conditionals_from_prob_logM_delta(
            analysis, prob_logM_delta_nn
        )

        model_logM = InterpolatedConditional1D(
            x_bins={
                "log_mass_total_s": analysis.binning["boundaries"]["log_mass_total_s"]
            },
            y_bins=None,
            cond=prob_logM,
            interpolation=interpolation,
        )
        model_delta = InterpolatedConditional1D(
            x_bins={
                "minus_log_mass_ratio": analysis.binning["boundaries"][
                    "minus_log_mass_ratio"
                ]
            },
            y_bins={
                "log_mass_total_s": analysis.binning["boundaries"]["log_mass_total_s"]
            },
            cond=prob_delta_given_logM,
            continuous_y_names=["log_mass_total_s"],
            interpolation=interpolation,
        )

        return model_logM, model_delta

    elif source_frame_masses_name in ["fourier_gp_2D_m1sq"]:
        prob_m1q_nn = construct_prob_nn_whitened_field_2D_m1sq(analysis, params)

        prob_m1s, prob_q_given_m1s = (
            construct_conditionals_from_prob_mass_1_s_mass_ratio(analysis, prob_m1q_nn)
        )

        model_m1s = InterpolatedConditional1D(
            x_bins={"mass_1_s": analysis.binning["boundaries"]["mass_1_s"]},
            y_bins=None,
            cond=prob_m1s,
            interpolation=interpolation,
        )
        model_q = InterpolatedConditional1D(
            x_bins={"mass_ratio": analysis.binning["boundaries"]["mass_ratio"]},
            y_bins={"mass_1_s": analysis.binning["boundaries"]["mass_1_s"]},
            cond=prob_q_given_m1s,
            continuous_y_names=["mass_1_s"],
            interpolation=interpolation,
        )

        return model_m1s, model_q

    else:
        raise NotImplementedError(
            f"Joint mass model '{source_frame_masses_name}' not implemented."
        )


def construct_mass_1_s_prob_nn(analysis, params):

    mass_1_s_name = analysis.get_distribution_name("mass_1_s")

    if mass_1_s_name == "power_law_peak":
        cond = construct_prob_nn_power_law_peak_1D(analysis, params)
    elif mass_1_s_name == "power_law_peak2":
        cond = construct_prob_nn_multipeak_1D(analysis, params)
    elif mass_1_s_name == "power_law_peak2_partial_windowed":
        cond = construct_prob_nn_multipeak_1D(analysis, params, partial_window=True)
    elif mass_1_s_name == "fourier_gp_1D":
        log_window = get_log_window_mass_s(analysis, params)
        log_prob = construct_log_prob_nn_whitened_field_1D(analysis, params, log_window)
        cond = jnp.exp(log_prob)
    else:
        raise NotImplementedError(f"Mass model {mass_1_s_name} not implemented.")

    return cond


def construct_mass_ratio_prob_nn(analysis, params):

    mass_ratio_name = analysis.get_distribution_name("mass_ratio")

    if mass_ratio_name == "mass_ratio_running_power_law_in_log":
        prob_mass_ratio = construct_running_power_law_prob_mass_ratio_nn(
            analysis, params
        )
    elif mass_ratio_name == "mass_ratio_truncated_gaussian":
        prob_mass_ratio = construct_prob_nn_truncated_gaussian_mass_ratio(
            analysis, params
        )
    else:
        raise NotImplementedError(
            "Only mass ratio running power law is implemented so far."
        )

    return prob_mass_ratio


def construct_conditionals_from_prob_logM_delta(analysis, prob_logM_delta_nn):
    """Factorize joint p(logM, delta) into p(logM) and p(delta | logM)."""

    edges_dict = analysis.binning["boundaries"]
    delta_dict = analysis.binning["deltas"]

    # Marginalize over minus_log_mass_ratio to get p(logM)
    prob_logM_nn = jnp.sum(
        prob_logM_delta_nn * delta_dict["minus_log_mass_ratio"][None, :], axis=-1
    )
    prob_logM = normalize_cond_interpolated_1d(
        x_edges=edges_dict["log_mass_total_s"],
        cond=prob_logM_nn,
    )

    # Conditional p(delta | logM) = p(logM, delta) / p(logM).  Sharp mass
    # windows can underflow an entire marginal row to zero; use a finite
    # fallback there so inactive rows do not poison interpolation with NaNs.
    prob_delta_given_logM_nn = _safe_conditional_from_joint(
        prob_logM_delta_nn,
        prob_logM_nn,
        delta_dict["minus_log_mass_ratio"],
    )

    # InterpolatedConditional1D expects shape (x_bins, y_bins) = (delta, logM)
    prob_delta_given_logM_nn_delta_logM = jnp.swapaxes(prob_delta_given_logM_nn, -1, -2)

    prob_delta_given_logM = normalize_cond_interpolated_1d(
        x_edges=edges_dict["minus_log_mass_ratio"],
        cond=prob_delta_given_logM_nn_delta_logM,
    )

    return prob_logM, prob_delta_given_logM


def construct_conditionals_from_prob_mass_1_s_mass_ratio(
    analysis, prob_mass_1_s_mass_ratio_nn_m1sq
):

    edges_dict = analysis.binning["boundaries"]
    delta_dict = analysis.binning["deltas"]

    prob_mass_1_s_nn_m1s = jnp.sum(
        prob_mass_1_s_mass_ratio_nn_m1sq * delta_dict["mass_ratio"][None, :], axis=-1
    )
    prob_m1s = normalize_cond_interpolated_1d(
        x_edges=edges_dict["mass_1_s"],
        cond=prob_mass_1_s_nn_m1s,
    )

    prob_mass_ratio_given_mass_1_s_nn_m1sq = _safe_conditional_from_joint(
        prob_mass_1_s_mass_ratio_nn_m1sq,
        prob_mass_1_s_nn_m1s,
        delta_dict["mass_ratio"],
    )

    # TODO implement named axes for this to avoid swapping axes
    prob_mass_ratio_given_mass_1_s_nn_qm1s = jnp.swapaxes(
        prob_mass_ratio_given_mass_1_s_nn_m1sq, -1, -2
    )

    prob_mass_ratio_given_mass_1_s_qm1s = normalize_cond_interpolated_1d(
        x_edges=edges_dict["mass_ratio"],
        cond=prob_mass_ratio_given_mass_1_s_nn_qm1s,
    )

    return prob_m1s, prob_mass_ratio_given_mass_1_s_qm1s
