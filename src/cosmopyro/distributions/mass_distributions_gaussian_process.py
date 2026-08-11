import jax.numpy as jnp

from ..field_utils import field
from ..utils.jax_utils import safe_sqrt, smooth_max
from .mass_distribution_parametrized import (
    get_log_window_mass_s,
)

__all__ = [
    "construct_log_prob_nn_whitened_field_1D",
    "construct_prob_nn_whitened_field_2D_logMdelta",
    "construct_prob_nn_whitened_field_2D_m1sq",
    "get_log_prob_from_field_prescription",
    "get_log_prob_imposed_prior_m1sq",
    "get_power_spectrum_2D_from_analysis_kwargs",
    "power_spectrum_1D",
    "power_spectrum_2D_constant_minus_cubic",
    "power_spectrum_2D_linear_minus_cubic",
    "power_spectrum_2D_quadratic_minus_quartic",
]


def _safe_normalization(norm):
    floor = jnp.asarray(1e-30, dtype=norm.dtype)
    return jnp.maximum(norm, floor)


def power_spectrum_1D(k, amplitude, cutoff):
    k_n = safe_sqrt(k**2) / cutoff
    normalization = 1 / cutoff
    return amplitude * normalization * k_n**2 / (1 + (k_n**6))


def power_spectrum_2D_constant_minus_cubic(
    k1, k2, amplitude, cutoff, relative_scale=1.0
):

    k_n_sq = (k1**2 + (k2 * relative_scale) ** 2) / cutoff**2
    shape = 1 / (1 + k_n_sq) ** 3

    normalization = relative_scale / cutoff**2

    return amplitude * normalization * shape


def power_spectrum_2D_linear_minus_cubic(k1, k2, amplitude, cutoff, relative_scale=1.0):

    # Work in k_n^2 so the only sqrt is safe_sqrt — jnp.sqrt(k1^2+k2^2) at the
    # origin has 0/0 gradients w.r.t. any parameter that scales a component
    # (relative_scale), silently poisoning gradients.
    k_n_sq = (k1**2 + (k2 * relative_scale) ** 2) / cutoff**2
    k_n = safe_sqrt(k_n_sq)

    # 2. The raw spectral shape
    shape = k_n / (1 + k_n_sq) ** 3

    normalization = relative_scale / cutoff**2

    return amplitude * normalization * shape


def power_spectrum_2D_quadratic_minus_quartic(
    k1, k2, amplitude, cutoff, relative_scale=1.0
):

    k_n_sq = (k1**2 + (k2 * relative_scale) ** 2) / cutoff**2

    # 2. The raw spectral shape
    shape = k_n_sq / (1 + k_n_sq**2) ** 2

    normalization = relative_scale / cutoff**2

    return amplitude * normalization * shape


def get_power_spectrum_2D_from_analysis_kwargs(kwargs_analysis=None):
    kwargs_analysis = kwargs_analysis or {}
    name = kwargs_analysis.get("distribution_names", {}).get(
        "source_frame_masses_power_spectrum",
        "linear_minus_cubic",
    )
    if name == "constant_minus_cubic":
        return power_spectrum_2D_constant_minus_cubic
    elif name == "linear_minus_cubic":
        return power_spectrum_2D_linear_minus_cubic
    elif name == "quadratic_minus_quartic":
        return power_spectrum_2D_quadratic_minus_quartic
    else:
        raise ValueError(f"Unknown power spectrum type: {name}")


# TODO rename and factorize this long function
def construct_log_prob_nn_whitened_field_1D(analysis, params, log_window):

    num_bins_mass_1_s = analysis.binning["deltas"]["mass_1_s"].shape[0]

    power_spectrum_of_k = power_spectrum_1D

    box_range = jnp.array([[0.0, 1.0]])
    box_shape_d = [num_bins_mass_1_s]
    field_instance = field.RealField(
        box_range_d=box_range,
        box_shape_d=box_shape_d,
        power_spectrum_of_k=power_spectrum_of_k,
        replace_FT_with_packing=False,
    )

    gaussian_F_whitened_spatial = params["mass_1_s"]["gaussian_F_whitened_spatial"]

    field_instance.set_gaussian_F_whitened_from_gaussian_F_whitened_spatial(
        gaussian_F_whitened_spatial
    )
    params_power_spectrum = dict(
        amplitude=params["mass_1_s"]["power_spectrum_amplitude"],
        cutoff=params["mass_1_s"]["power_spectrum_cutoff"],
    )
    field_instance.compute_gaussian_F_spatial_from_gaussian_F_whitened(
        power_spectrum_kwargs=params_power_spectrum
    )

    log_prob_gaussian = field_instance.gaussian_F_spatial - smooth_max(
        field_instance.gaussian_F_spatial
    )

    log_prior = jnp.log(analysis.binning["centers"]["mass_1_s"]) * params[
        "mass_1_s"
    ].get("alpha_0", 0.0)

    # add window
    log_prob_gaussian = log_prob_gaussian + log_window + log_prior

    log_prob_gaussian_norm = jnp.log(
        jnp.sum(jnp.exp(log_prob_gaussian) * analysis.binning["deltas"]["mass_1_s"])
    )
    log_prob_gaussian = log_prob_gaussian - log_prob_gaussian_norm

    return log_prob_gaussian


def get_log_prob_from_field_prescription(analysis, gaussian_field):
    kwargs_analysis = getattr(analysis, "kwargs_analysis", {}) or {}
    name = kwargs_analysis.get("distribution_names", {}).get(
        "field_to_log_prob_prescription",
        "linear",
    )
    if name == "linear":
        log_prob_gaussian = gaussian_field
    elif name == "log_quadratic":
        log_prob_gaussian = jnp.log(gaussian_field**2 + 1e-14)
    else:
        raise ValueError(f"Unknown field to log prob prescription: {name}")
    return log_prob_gaussian


def construct_prob_nn_whitened_field_2D_logMdelta(analysis, params):
    """
    Build a 2D GRF on (log_mass_total_s, minus_log_mass_ratio) coordinates
    with exchange symmetry enforced by mirroring the white noise in the
    minus_log_mass_ratio direction before FFT.

    The GRF grid covers [logM_min, logM_max] x [-delta_max, +delta_max] where
    delta = minus_log_mass_ratio. The physical region is delta >= 0 (m1 >= m2).
    The symmetry phi(logM, delta) = phi(logM, -delta) is exact by construction.

    Returns prob_nn on the (log_mass_total_s, minus_log_mass_ratio >= 0) grid.
    """

    num_bins_logM = analysis.binning["deltas"]["log_mass_total_s"].shape[0]
    num_bins_delta = analysis.binning["deltas"]["minus_log_mass_ratio"].shape[0]
    num_bins_delta_full = 2 * num_bins_delta

    # Window: compute m1_s and m2_s at each (logM, delta) grid point
    logM_centers = analysis.binning["centers"]["log_mass_total_s"]
    delta_centers = analysis.binning["centers"]["minus_log_mass_ratio"]
    # m1_s = exp(logM) / (1 + exp(-delta)),  m2_s = exp(logM) / (1 + exp(delta))
    m1_s_grid = jnp.exp(logM_centers[:, None]) / (1 + jnp.exp(-delta_centers[None, :]))
    m2_s_grid = jnp.exp(logM_centers[:, None]) / (1 + jnp.exp(delta_centers[None, :]))
    mass_ratio_grid = jnp.exp(-delta_centers[None, :] * jnp.ones((num_bins_logM, 1)))

    mass_params_key = "source_frame_masses"
    mass_params = params[mass_params_key]

    log_window_m1 = get_log_window_mass_s(
        analysis, params, bins_mass_s=m1_s_grid, mass_params_key=mass_params_key
    )
    log_window_m2 = get_log_window_mass_s(
        analysis, params, bins_mass_s=m2_s_grid, mass_params_key=mass_params_key
    )
    log_window_logM_delta = log_window_m1 + log_window_m2

    # Build the GRF on the full symmetric grid [logM] x [-delta_max, +delta_max]
    box_range = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    box_shape_d = [num_bins_logM, num_bins_delta_full]
    power_spectrum_of_k = get_power_spectrum_2D_from_analysis_kwargs(
        getattr(analysis, "kwargs_analysis", None)
    )
    field_instance = field.RealField(
        box_range_d=box_range,
        box_shape_d=box_shape_d,
        power_spectrum_of_k=power_spectrum_of_k,
        replace_FT_with_packing=False,
    )

    if "gaussian_F_whitened_spatial_marginal" in mass_params:
        field_instance_marginal = field.RealField(
            box_range_d=jnp.array([[0.0, 1.0]]),
            box_shape_d=[num_bins_logM],
            power_spectrum_of_k=power_spectrum_1D,
        )

        gaussian_F_whitened_spatial_marginal = mass_params[
            "gaussian_F_whitened_spatial_marginal"
        ]
        field_instance_marginal.set_gaussian_F_whitened_from_gaussian_F_whitened_spatial(
            gaussian_F_whitened_spatial_marginal
        )
        params_power_spectrum_marginal = dict(
            amplitude=mass_params["power_spectrum_amplitude_marginal"],
            cutoff=mass_params["power_spectrum_cutoff_marginal"],
        )
        field_instance_marginal.compute_gaussian_F_spatial_from_gaussian_F_whitened(
            power_spectrum_kwargs=params_power_spectrum_marginal
        )
        log_prob_marginal = field_instance_marginal.gaussian_F_spatial
        log_prob_marginal = log_prob_marginal - smooth_max(log_prob_marginal)
    else:
        log_prob_marginal = jnp.zeros(num_bins_logM)

    # White noise has shape [N_logM, N_delta_half] — the independent parameters.
    # Mirror to enforce exchange symmetry: phi(logM, delta) = phi(logM, -delta).
    gaussian_F_whitened_half = mass_params["gaussian_F_whitened_spatial"]
    gaussian_F_whitened_full = jnp.concatenate(
        [
            jnp.flip(gaussian_F_whitened_half, axis=-1),
            gaussian_F_whitened_half,
        ],
        axis=-1,
    )

    field_instance.set_gaussian_F_whitened_from_gaussian_F_whitened_spatial(
        gaussian_F_whitened_full
    )
    params_power_spectrum = dict(
        amplitude=mass_params["power_spectrum_amplitude"],
        cutoff=mass_params["power_spectrum_cutoff"],
        relative_scale=mass_params.get(
            "power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio",
            1.0,
        ),
    )
    field_instance.compute_gaussian_F_spatial_from_gaussian_F_whitened(
        power_spectrum_kwargs=params_power_spectrum
    )

    log_prob_gaussian = get_log_prob_from_field_prescription(
        analysis, field_instance.gaussian_F_spatial[:, num_bins_delta:]
    )
    log_prob_gaussian = log_prob_gaussian - smooth_max(log_prob_gaussian)

    # optional prior
    ref_power_law_m1s = mass_params.get("power_law_reference_mass_1_s", 0.0)
    ref_power_law_q = mass_params.get("power_law_reference_mass_ratio", 0.0)
    log_prior = (
        jnp.log(m1_s_grid) * ref_power_law_m1s
        + jnp.log(mass_ratio_grid) * ref_power_law_q
    )

    # Apply mass window
    log_prob_gaussian_masked = (
        log_prob_gaussian
        + log_window_logM_delta
        + log_prior
        + log_prob_marginal[:, None]
    )

    # Normalize over (logM, delta >= 0)
    delta_logM = analysis.binning["deltas"]["log_mass_total_s"]
    delta_delta = analysis.binning["deltas"]["minus_log_mass_ratio"]
    normalization = jnp.sum(
        jnp.exp(log_prob_gaussian_masked) * delta_logM[:, None] * delta_delta[None, :]
    )
    norm_safe = _safe_normalization(normalization)
    log_prob_gaussian_masked = log_prob_gaussian_masked - jnp.log(norm_safe)

    prob_nn = jnp.exp(log_prob_gaussian_masked)

    return prob_nn


def construct_prob_nn_whitened_field_2D_m1sq(analysis, params):
    """
    Build a 2D GRF on (mass_1_s, mass_ratio) coordinates.

    No exchange symmetry mirroring is needed since q in (0, 1]
    already enforces m1 >= m2. The window is applied on both m1 and m2.

    Returns prob_nn on the (mass_1_s, mass_ratio) grid.
    """

    num_bins_m1 = analysis.binning["deltas"]["mass_1_s"].shape[0]
    num_bins_q = analysis.binning["deltas"]["mass_ratio"].shape[0]

    # Window: compute m2_s at each (m1, q) grid point
    m1_centers = analysis.binning["centers"]["mass_1_s"]
    q_centers = analysis.binning["centers"]["mass_ratio"]
    m2_grid = m1_centers[:, None] * q_centers[None, :]

    mass_params_key = "source_frame_masses"
    mass_params = params[mass_params_key]

    log_window_m1 = get_log_window_mass_s(
        analysis, params, bins_mass_s=m1_centers, mass_params_key=mass_params_key
    )
    log_window_m2 = get_log_window_mass_s(
        analysis, params, bins_mass_s=m2_grid, mass_params_key=mass_params_key
    )
    log_window = log_window_m1[:, None] + log_window_m2

    # Build the GRF
    box_range = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    box_shape_d = [num_bins_m1, num_bins_q]

    power_spectrum_of_k = get_power_spectrum_2D_from_analysis_kwargs(
        getattr(analysis, "kwargs_analysis", None)
    )
    field_instance = field.RealField(
        box_range_d=box_range,
        box_shape_d=box_shape_d,
        power_spectrum_of_k=power_spectrum_of_k,
        replace_FT_with_packing=False,
    )

    gaussian_F_whitened = mass_params["gaussian_F_whitened_spatial"]
    field_instance.set_gaussian_F_whitened_from_gaussian_F_whitened_spatial(
        gaussian_F_whitened
    )

    params_power_spectrum = dict(
        amplitude=mass_params["power_spectrum_amplitude"],
        cutoff=mass_params["power_spectrum_cutoff"],
        relative_scale=mass_params.get(
            "power_spectrum_relative_scale_mass_1_s_to_mass_ratio",
            1.0,
        ),
    )
    field_instance.compute_gaussian_F_spatial_from_gaussian_F_whitened(
        power_spectrum_kwargs=params_power_spectrum
    )

    log_prob_gaussian = field_instance.gaussian_F_spatial
    log_prob_gaussian = log_prob_gaussian - smooth_max(log_prob_gaussian)

    ref_power_law_m1s = mass_params.get("power_law_reference_mass_1_s", 0.0)
    ref_power_law_q = mass_params.get("power_law_reference_mass_ratio", 0.0)
    log_prior = (
        jnp.log(m1_centers)[:, None] * ref_power_law_m1s
        + jnp.log(q_centers)[None, :] * ref_power_law_q
    )

    # Apply mass window
    log_prob_gaussian_masked = log_prob_gaussian + log_window + log_prior

    # Normalize over (m1, q)
    delta_m1 = analysis.binning["deltas"]["mass_1_s"]
    delta_q = analysis.binning["deltas"]["mass_ratio"]
    normalization = jnp.sum(
        jnp.exp(log_prob_gaussian_masked) * delta_m1[:, None] * delta_q[None, :]
    )
    norm_safe = _safe_normalization(normalization)
    log_prob_gaussian_masked = log_prob_gaussian_masked - jnp.log(norm_safe)

    prob_nn = jnp.exp(log_prob_gaussian_masked)

    return prob_nn


def get_log_prob_imposed_prior_m1sq(analysis, params, log_window_m1sq):

    q_centers = analysis.binning["centers"]["mass_ratio"]

    beta_ref = 1.0

    mean_trend_log_prob_mass_ratio_m1sq = (
        jnp.log(q_centers)[None, :] * beta_ref + log_window_m1sq
    )
    log_normalization_m1s = jnp.logaddexp.reduce(
        mean_trend_log_prob_mass_ratio_m1sq, axis=-1
    )
    mean_trend_log_prob_mass_ratio_masked_m1sq = (
        mean_trend_log_prob_mass_ratio_m1sq - log_normalization_m1s[:, None]
    )
    return mean_trend_log_prob_mass_ratio_masked_m1sq
