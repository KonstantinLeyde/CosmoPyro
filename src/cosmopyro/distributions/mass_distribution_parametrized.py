import jax
import jax.numpy as jnp

from ..utils.jax_utils import log_smooth_heaviside_window
from .grid_distributions import (
    normalize_cond_interpolated_1d,
)

__all__ = [
    "LOW_MASS_SMOOTHINGS",
    "LVK_SMOOTHING_FLOOR",
    "SMOOTHING_SCALE",
    "complete_mass_bins_for_mass_ratio_models",
    "construct_prob_nn_multipeak_1D",
    "construct_prob_nn_power_law_peak_1D",
    "construct_prob_nn_truncated_gaussian_mass_ratio",
    "construct_running_power_law_prob_mass_ratio_nn",
    "get_log_window_mass_ratio",
    "get_log_window_mass_s",
    "get_mass_delta_m2",
    "get_mass_min",
    "log_low_mass_smoothing",
    "log_smoothing_low_mass_gaussian",
    "log_smoothing_low_mass_lvk",
]

SMOOTHING_SCALE = 0.007

## P(mass_1_s) models


def construct_prob_nn_power_law_peak_1D(analysis, params, *, smoothing):
    """
    Power-law + Gaussian peak model for P(mass_1_s).

    Parameters
    ----------
    analysis : Analysis
        The analysis object containing binning information.
    params : dict
        Dictionary containing the parameters for the mass distribution model.
    smoothing : str
        Low-mass turn-on, a key of ``LOW_MASS_SMOOTHINGS`` ("gaussian" or "lvk").

    Returns
    -------
    prob_mass_1_s: jnp.ndarray
        The probability distribution P(mass_1_s) evaluated at the bin centers.

    """

    m1 = analysis.binning["centers"]["mass_1_s"]

    d = params["mass_1_s"]

    alpha = d["alpha"]
    mmin = d["mmin"]
    mmax = d["mmax"]
    mu_g = d["mu_g"]
    sigma_g = d["sigma_g"]
    lambda_peak = d["lambda_peak"]
    delta_m = d["delta_m"]

    log_P = _log_prob_TPL(m1, alpha, mmin, mmax)
    log_G = _log_prob_G(m1, mu_g, sigma_g)

    # NOTE: icarogw applies this window solely to the power law component
    # (but gwpopulation applies it to the entire mixture)
    # here we apply it to the entire mixture, which is consistent with the gwpopulation implementation
    log_window = log_smooth_heaviside_window(
        m1, mmin, mmax, SMOOTHING_SCALE * mmin, SMOOTHING_SCALE * mmax
    )
    log_mixture = jnp.logaddexp(
        jnp.log(1.0 - lambda_peak) + log_P, jnp.log(lambda_peak) + log_G
    )

    log_val = (
        log_mixture
        + log_low_mass_smoothing(smoothing, m1, mmin, delta_m)
        + log_window
    )

    return jnp.exp(log_val)


def construct_prob_nn_multipeak_1D(analysis, params, *, smoothing, partial_window=False):
    """
    Power-law + two Gaussian peaks model for P(mass_1_s), with optional partial windowing.
    If partial_window is True, solely the power-law component is windowed, while the Gaussian peaks are not.
    This corresponds to the icarogw implementation.

    If partial_window is False, the entire mixture is windowed, which corresponds to the gwpopulation implementation.

    The low-mass turn-on ``smoothing`` (a key of ``LOW_MASS_SMOOTHINGS``, "gaussian" or "lvk")
    is applied to the entire mixture in both cases.

    Returns
    -------
    prob_mass_1_s: jnp.ndarray
        The probability distribution P(mass_1_s) evaluated at the bin centers.

    """

    m1 = analysis.binning["centers"]["mass_1_s"]

    d = params["mass_1_s"]

    alpha = d["alpha"]
    mmin = d["mmin"]
    mmax = d["mmax"]
    mu_g_low = d["mu_g_low"]
    mu_g_high = d["mu_g_high"]
    sigma_g_low = d["sigma_g_low"]
    sigma_g_high = d["sigma_g_high"]
    lambda_g_low = d["lambda_g_low"]
    lambda_g = d["lambda_g"]
    delta_m = d["delta_m"]

    log_P = _log_prob_TPL(m1, alpha, mmin, mmax)
    log_G_low = _log_prob_G(m1, mu_g_low, sigma_g_low)
    log_G_high = _log_prob_G(m1, mu_g_high, sigma_g_high)

    log_window = log_smooth_heaviside_window(
        m1, mmin, mmax, SMOOTHING_SCALE * mmin, SMOOTHING_SCALE * mmax
    )

    log_w_pl = jnp.log(1.0 - lambda_g)
    log_w_low = jnp.log(lambda_g) + jnp.log(lambda_g_low)
    log_w_high = jnp.log(lambda_g) + jnp.log(1 - lambda_g_low)

    log_prob_peaks = jnp.logaddexp(log_w_high + log_G_high, log_w_low + log_G_low)

    if partial_window:
        log_pl_windowed = log_w_pl + log_P + log_window
        log_mixture = jnp.logaddexp(log_pl_windowed, log_prob_peaks)
        log_val = log_mixture + log_low_mass_smoothing(smoothing, m1, mmin, delta_m)
    else:
        log_mixture = jnp.logaddexp(
            log_prob_peaks,
            log_w_pl + log_P,
        )
        log_val = (
            log_mixture
            + log_low_mass_smoothing(smoothing, m1, mmin, delta_m)
            + log_window
        )

    return jnp.exp(log_val)


## P(mass_1_s, mass_ratio) models


def get_mass_min(params, mass_params_key="mass_1_s"):
    """
    Get the minimum mass for a given mass distribution model, based on the provided parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters for the mass distribution model.
    mass_params_key : str, optional
        Key to access the mass parameters in the params dictionary. Default is 'mass_1_s

    Returns
    -------
    mass_min : float
        The minimum mass for the specified mass distribution model.

    """

    mass_params = params[mass_params_key]
    mass_min = mass_params.get("mass_min", mass_params.get("mmin", None))
    if mass_min is None:
        raise NotImplementedError(
            f"Mass with params = {params.keys()} not recognized for getting mass_min."
        )
    return mass_min


def get_mass_delta_m2(params):
    """
    Get the delta_m2 smoothing parameter for a given mass distribution model, based on the provided parameters.
    If 'sigma_mass_cutoff_mass_2' is not specified in the 'mass_ratio' parameters, it falls back to using
    'delta_m' from the 'mass_1_s' parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters for the mass distribution model.

    Returns
    -------
    delta_m2 : float
        The delta_m2 smoothing parameter for the specified mass distribution model.

    """

    # if sigma_mass_cutoff_mass_2 is not specified, use delta_m
    delta_m2 = params["mass_ratio"].get(
        "sigma_mass_cutoff_mass_2", params["mass_1_s"].get("delta_m", None)
    )
    if delta_m2 is None:
        raise NotImplementedError(
            f"Mass with params = {params.keys()} not recognized for getting delta_m2."
        )
    return delta_m2


def complete_mass_bins_for_mass_ratio_models(
    analysis, centers_mass_1_s=None, centers_mass_ratio=None, centers_mass_2_s=None
):
    """
    Complete the mass bins for mass ratio models, given either centers_mass_1_s and centers_mass_ratio, or centers_mass_1_s and centers_mass_2_s.
    If centers_mass_1_s is None, use the default binning from analysis.binning
    If centers_mass_ratio is None, use the default binning from analysis.binning

    The returned m1s, q, and m2s are 2D arrays with shape (len(centers_mass_ratio), len(centers_mass_1_s))
    and (len(centers_mass_2_s), len(centers_mass_1_s)) respectively, where each row corresponds to a fixed
    mass ratio or mass_2_s, and each column corresponds to a fixed mass_1_s.

    """

    if centers_mass_1_s is None:
        centers_mass_1_s = analysis.binning["centers"]["mass_1_s"]

    m1s = centers_mass_1_s[None, :]

    if centers_mass_ratio is not None and centers_mass_2_s is not None:
        raise ValueError("Provide only one of centers_mass_ratio or centers_mass_2_s.")

    if centers_mass_ratio is not None:
        q = centers_mass_ratio[:, None]
    elif centers_mass_2_s is not None:
        q = centers_mass_2_s[:, None] / m1s
    else:
        q = analysis.binning["centers"]["mass_ratio"][:, None]

    m2s = m1s * q
    return m1s, q, m2s


def construct_running_power_law_prob_mass_ratio_nn(
    analysis,
    params,
    *,
    smoothing,
    centers_mass_1_s=None,
    centers_mass_ratio=None,
    centers_mass_2_s=None,
):
    """
    Construct the non-normalized probability distribution for the mass ratio, P(mass_ratio | mass_1_s), using a running power-law model.
    beta(m1s) = beta_0 + beta_1 * (log(m1s) - log(ms_ref))

    Parameters
    ----------
    analysis : Analysis
        The analysis object containing binning information.
    params : dict
        Dictionary containing the parameters for the mass distribution model.
    smoothing : str
        Low-mass turn-on in mass_2_s, a key of ``LOW_MASS_SMOOTHINGS`` ("gaussian" or "lvk").
    centers_mass_1_s : array-like, optional
        The centers of the mass_1_s bins. If None, use the default binning from analysis.binning.
    centers_mass_ratio : array-like, optional
        The centers of the mass_ratio bins. If None, use the default binning from analysis.binning.
    centers_mass_2_s : array-like, optional
        The centers of the mass_2_s bins. If None, use the default binning from analysis.binning.

    Returns
    -------
    prob_mass_ratio : jnp.ndarray
        The non-normalized probability distribution P(mass_ratio | mass_1_s) evaluated at the bin centers.
        Shape is (len(centers_mass_ratio), len(centers_mass_1_s)) if centers_mass_ratio is provided, or
        (len(centers_mass_2_s), len(centers_mass_1_s)) if centers_mass_2_s is provided.

    """

    m1s, q, m2s = complete_mass_bins_for_mass_ratio_models(
        analysis, centers_mass_1_s, centers_mass_ratio, centers_mass_2_s
    )

    b0 = params["mass_ratio"]["beta_0"]
    b1 = params["mass_ratio"]["beta_1"]

    mass_min = get_mass_min(params)
    sigma_mass_cutoff_mass_2 = get_mass_delta_m2(params)

    log_smooth_cutoff = log_low_mass_smoothing(
        smoothing, m2s, mass_min, sigma_mass_cutoff_mass_2
    )

    ms_ref = params["mass_ratio"]["mass_ratio_running_zero_point"]
    b = b0 + b1 * (jnp.log(m1s) - jnp.log(ms_ref))

    prob_nn_mass_ratio = q**b

    # apply cutoff
    prob_nn_mass_ratio = jnp.exp(log_smooth_cutoff) * prob_nn_mass_ratio

    prob_mass_ratio = normalize_cond_interpolated_1d(
        analysis.binning["boundaries"]["mass_ratio"], prob_nn_mass_ratio
    )

    return prob_mass_ratio


def construct_prob_nn_truncated_gaussian_mass_ratio(
    analysis,
    params,
    *,
    smoothing,
    centers_mass_1_s=None,
    centers_mass_ratio=None,
    centers_mass_2_s=None,
):
    """
    Construct the non-normalized probability distribution for the mass ratio, P(mass_ratio | mass_1_s), using a truncated Gaussian model.
    The Gaussian is truncated at q=0 and q=1, and is smoothed at the low-mass end in mass_2_s.

    Parameters
    ----------
    analysis : Analysis
        The analysis object containing binning information.
    params : dict
        Dictionary containing the parameters for the mass distribution model.
    smoothing : str
        Low-mass turn-on in mass_2_s, a key of ``LOW_MASS_SMOOTHINGS`` ("gaussian" or "lvk").
    centers_mass_1_s : array-like, optional
        The centers of the mass_1_s bins. If None, use the default binning from analysis.binning.
    centers_mass_ratio : array-like, optional
        The centers of the mass_ratio bins. If None, use the default binning from analysis.binning.
    centers_mass_2_s : array-like, optional
        The centers of the mass_2_s bins. If None, use the default binning from analysis.binning.

    Returns
    -------
    prob_mass_ratio : jnp.ndarray
        The non-normalized probability distribution P(mass_ratio | mass_1_s) evaluated at the bin centers.
        Shape is (len(centers_mass_ratio), len(centers_mass_1_s)) if centers_mass_ratio is provided, or
        (len(centers_mass_2_s), len(centers_mass_1_s)) if centers_mass_2_s is provided.

    """

    _, q, m2s = complete_mass_bins_for_mass_ratio_models(
        analysis, centers_mass_1_s, centers_mass_ratio, centers_mass_2_s
    )

    mu = params["mass_ratio"]["mu_mass_ratio"]
    sigma = params["mass_ratio"]["sigma_mass_ratio"]

    mass_min = get_mass_min(params)
    sigma_mass_cutoff_mass_2 = get_mass_delta_m2(params)

    log_smooth_cutoff = log_low_mass_smoothing(
        smoothing, m2s, mass_min, sigma_mass_cutoff_mass_2
    )

    prob_nn_mass_ratio = jnp.exp(-0.5 * ((q - mu) / sigma) ** 2) / (
        sigma * jnp.sqrt(2 * jnp.pi)
    )
    prob_nn_mass_ratio = jnp.exp(log_smooth_cutoff) * prob_nn_mass_ratio

    prob_mass_ratio = normalize_cond_interpolated_1d(
        analysis.binning["boundaries"]["mass_ratio"], prob_nn_mass_ratio
    )

    return prob_mass_ratio


# Below modified from Chimera
def _log_prob_TPL(x, alpha, mmin, mmax):
    log_window = log_smooth_heaviside_window(
        x, mmin, mmax, SMOOTHING_SCALE * mmin, SMOOTHING_SCALE * mmax
    )
    term_low = mmin ** (1 - alpha) / (1 - alpha)
    term_high = mmax ** (1 - alpha) / (1 - alpha)
    log_norm_cost = jnp.log(term_high - term_low)
    return -alpha * jnp.log(x) - log_norm_cost + log_window


def _log_prob_G(x, mu, sigma):
    return (
        -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - (x - mu) ** 2 / (2.0 * sigma**2)
    )


# End of modified from Chimera


def log_smoothing_low_mass_gaussian(
    ms: jnp.ndarray,
    mmin: float,
    delta_m: float,
) -> jnp.ndarray:
    r"""
    Gaussian low-mass turn-on: a Gaussian fall-off below a threshold and S ~ 1 above it,

    .. math::

        \log S(m) = -c\, \mathrm{softplus}(t)^2, \qquad
        t = \frac{m_\mathrm{min} + \delta_m + a - m}{b\, \delta_m}.

    It is never exactly zero, also below mmin. The constants a, b, c are tuned so that it
    roughly follows the LVK turn-on (``log_smoothing_low_mass_lvk``), i.e. S ~ 0 at
    m = mmin and S ~ 1 at m = mmin + delta_m.

    Parameters
    ----------
    ms : jnp.ndarray
        The mass values at which to evaluate the smoothing function.
    mmin : float
        The minimum mass below which the smoothing function is approximately 0.
    delta_m : float
        The mass range over which the smoothing function transitions from 0 to 1.

    """

    # numbers to approximately match the LVK exponential smoothing
    a, b, c = -0.16256875, 0.07860472, 0.02496621

    # The point where the low-mass turn-on/cutoff finishes
    threshold = mmin + delta_m + a

    # Normalized distance *below* the threshold
    # If m1 > threshold, t is negative. If m1 < threshold, t is positive.
    t = (threshold - ms) / b / delta_m

    # jax.nn.softplus(t) smoothly approaches t for large positive values,
    # and smoothly approaches 0 for negative values.
    # It replaces: jnp.maximum(0.0, t)
    t_smooth = jax.nn.softplus(t)

    # Quadratic decay in log-space: log(p) = -0.5 * t_smooth^2
    return -c * (t_smooth**2)


# Value of the LVK turn-on S below mmin (and wherever the exact S is smaller), so that
# log S stays finite on the mass grid: a hard zero would give -inf grid log-densities and
# NaN gradients.
LVK_SMOOTHING_FLOOR = 1e-30


def log_smoothing_low_mass_lvk(
    ms: jnp.ndarray,
    mmin: float,
    delta_m: float,
) -> jnp.ndarray:
    r"""
    LVK low-mass turn-on (Talbot & Thrane 2018, as in gwpopulation and icarogw), in log space:

    .. math::

        S(m) = \begin{cases}
            0 & m < m_\mathrm{min} \\
            \left[1 + \exp\left(\frac{\delta_m}{m'} + \frac{\delta_m}{m' - \delta_m}\right)\right]^{-1}
                & m_\mathrm{min} \le m < m_\mathrm{min} + \delta_m \\
            1 & m \ge m_\mathrm{min} + \delta_m
        \end{cases}, \qquad m' = m - m_\mathrm{min},

    except that S is floored at ``LVK_SMOOTHING_FLOOR``: log S = max(log S_exact, log floor).
    The result is finite and continuous everywhere, and its gradient is finite (zero on the floor).

    Parameters
    ----------
    ms : jnp.ndarray
        The mass values at which to evaluate the smoothing function.
    mmin : float
        The minimum mass; S is at the floor for ms <= mmin.
    delta_m : float
        The width of the turn-on; S = 1 for ms >= mmin + delta_m.

    """

    log_floor = jnp.log(LVK_SMOOTHING_FLOOR)
    m_prime = ms - mmin
    inside = (m_prime > 0.0) & (m_prime < delta_m)

    # Evaluate the exact expression only inside the turn-on; elsewhere use a harmless point,
    # so that neither the value nor the gradient of the unused branch is inf / NaN.
    m_prime_safe = jnp.where(inside, m_prime, 0.5 * delta_m)
    exponent = delta_m / m_prime_safe + delta_m / (m_prime_safe - delta_m)
    log_s_inside = -jax.nn.softplus(exponent)

    log_s = jnp.where(
        inside, log_s_inside, jnp.where(m_prime >= delta_m, 0.0, log_floor)
    )
    return jnp.maximum(log_s, log_floor)


LOW_MASS_SMOOTHINGS = {
    "gaussian": log_smoothing_low_mass_gaussian,
    "lvk": log_smoothing_low_mass_lvk,
}


def log_low_mass_smoothing(smoothing, ms, mmin, delta_m):
    """log S(ms; mmin, delta_m) for the low-mass turn-on ``smoothing`` ("gaussian" or "lvk")."""
    if smoothing not in LOW_MASS_SMOOTHINGS:
        raise ValueError(
            f"Unknown low-mass smoothing {smoothing!r}; expected one of {list(LOW_MASS_SMOOTHINGS)}."
        )
    return LOW_MASS_SMOOTHINGS[smoothing](ms, mmin, delta_m)


def get_log_window_mass_s(
    analysis, params, bins_mass_s=None, mass_params_key="mass_1_s"
):

    if bins_mass_s is None:
        bins_mass_s = analysis.binning["centers"]["mass_1_s"]

    mass_params = params[mass_params_key]
    mass_min, mass_max = mass_params["mass_min"], mass_params["mass_max"]
    sigma_low = mass_params["sigma_low_fractional"] * mass_min
    sigma_high = mass_params["sigma_high_fractional"] * mass_max

    log_window = log_smooth_heaviside_window(
        bins_mass_s, mass_min, mass_max, sigma_low, sigma_high
    )

    return log_window


def get_log_window_mass_ratio(analysis, params):

    m1s = analysis.binning["centers"]["mass_1_s"]
    q = analysis.binning["centers"]["mass_ratio"]

    m2s = m1s[:, None] * q[None, :]

    return get_log_window_mass_s(analysis, params, bins_mass_s=m2s)
