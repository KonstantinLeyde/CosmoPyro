import operator
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node, tree_map, tree_reduce

__all__ = [
    "SQRT2",
    "check_normalization",
    "compute_centers_and_delta_from_array",
    "cubic_hermite_interp",
    "get_jnp_array_or_None",
    "get_penalty_factor_relative_variance",
    "get_relative_variance",
    "group_sizes_are_uniform",
    "handle_nonfinite_log_probs",
    "histdd",
    "linear_array_from_dict",
    "log_estimator_and_variance",
    "log_estimator_and_variance_stacked",
    "log_smooth_heaviside_erf",
    "log_smooth_heaviside_window",
    "logistic",
    "logmeanexp",
    "logmeanexp_last_axis",
    "logsumexp_last_axis",
    "make_segment_ids_from_group_sizes",
    "normalize_probs_over_axes",
    "rbf_kernel_1d",
    "safe_sqrt",
    "set_nan_and_neg_inf_to_min_plus_offset",
    "smooth_heaviside_erf",
    "smooth_interp",
    "smooth_max",
    "soft_clamp_low",
    "subtract_max",
    "subtract_mean",
    "sum_log_probs",
]


# --- Global JAX Registrations ---
# This runs once, the first time this module is imported anywhere.
# Guarded against double-registration so Jupyter autoreload doesn't crash.
#
# Keys are sorted so the treedef is invariant to attribute insertion order.
# Without sorting, two SimpleNamespaces loaded from HDF5 files with different
# internal key ordering would produce different treedefs, breaking tree_map
# across them and causing spurious jit retraces.
def _flatten_simple_namespace(node):
    items = sorted(vars(node).items())
    children = [v for _, v in items]
    aux_data = tuple(k for k, _ in items)
    return children, aux_data


def _unflatten_simple_namespace(aux_data, children):
    return SimpleNamespace(**dict(zip(aux_data, children)))


try:
    register_pytree_node(
        SimpleNamespace,
        _flatten_simple_namespace,
        _unflatten_simple_namespace,
    )
except ValueError:
    # Already registered (e.g. due to autoreload re-executing this module).
    pass

# ---------------------------------------------------------------------------
# Smooth interpolation utilities (C2-continuous cubic Hermite spline)
# ---------------------------------------------------------------------------


def _cubic_hermite_weights(t):
    """Hermite basis polynomials for t in [0,1]. Returns (h00, h10, h01, h11)."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00, h10, h01, h11


def _catmull_rom_slopes(x, y):
    """Catmull-Rom finite-difference slopes for monotone cubic Hermite.

    Uses three-point finite differences in the interior and one-sided
    differences at the boundaries. This gives C1-continuous interpolation
    that is C2-smooth away from the endpoints.
    """
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    slopes_secant = dy / dx

    # Interior: weighted harmonic-mean-like Catmull-Rom tangent
    m = jnp.empty_like(y)
    m = m.at[1:-1].set(0.5 * (slopes_secant[:-1] + slopes_secant[1:]))
    m = m.at[0].set(slopes_secant[0])
    m = m.at[-1].set(slopes_secant[-1])
    return m


@partial(jax.jit, static_argnums=())
def cubic_hermite_interp(x_query, x_knots, y_knots, slopes):
    """Evaluate a cubic Hermite spline at query points.

    Parameters
    ----------
    x_query : array  – points at which to interpolate
    x_knots : (N,) array  – strictly increasing knot positions
    y_knots : (N,) array  – function values at knots
    slopes  : (N,) array  – first derivatives at knots (e.g. from _catmull_rom_slopes)

    Returns
    -------
    y_query : array  – interpolated values (same shape as x_query)
    """
    # Find left-knot index for each query point
    idx = jnp.searchsorted(x_knots, x_query, side="right") - 1
    idx = jnp.clip(idx, 0, x_knots.shape[0] - 2)

    x0 = x_knots[idx]
    x1 = x_knots[idx + 1]
    y0 = y_knots[idx]
    y1 = y_knots[idx + 1]
    m0 = slopes[idx]
    m1 = slopes[idx + 1]

    dx = x1 - x0
    t = (x_query - x0) / dx

    h00, h10, h01, h11 = _cubic_hermite_weights(t)
    return h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1


def smooth_interp(x_query, x_knots, y_knots, left=None, right=None, **kwargs):
    """Drop-in replacement for jnp.interp with C1-continuous cubic Hermite spline.

    Computes Catmull-Rom slopes automatically. For points outside the knot range:
    - left/right=None: constant extrapolation (boundary value), like jnp.interp default.
    - left/right='extrapolate': linear extrapolation using boundary slope.
    """
    slopes = _catmull_rom_slopes(x_knots, y_knots)
    # Clamp queries into the knot range before interpolation to avoid
    # wild cubic extrapolation / NaN. Use double-where to block gradients.
    lo, hi = x_knots[0], x_knots[-1]
    oob_lo = x_query < lo
    oob_hi = x_query > hi
    x_safe = jnp.where(oob_lo | oob_hi, 0.5 * (lo + hi), x_query)
    result = cubic_hermite_interp(x_safe, x_knots, y_knots, slopes)

    if left == "extrapolate":
        # Linear extrapolation below the range using the boundary slope
        extrap_lo = y_knots[0] + slopes[0] * (x_query - lo)
        result = jnp.where(oob_lo, extrap_lo, result)
    else:
        result = jnp.where(oob_lo, y_knots[0], result)

    if right == "extrapolate":
        extrap_hi = y_knots[-1] + slopes[-1] * (x_query - hi)
        result = jnp.where(oob_hi, extrap_hi, result)
    else:
        result = jnp.where(oob_hi, y_knots[-1], result)

    return result


# ---------------------------------------------------------------------------
# Safe sqrt (finite gradient at zero)
# ---------------------------------------------------------------------------


def safe_sqrt(x):
    """sqrt with a finite gradient at x=0.

    jnp.sqrt has an infinite derivative at 0; combined with a vanishing
    upstream factor (e.g., the k=0 mode of a physical power spectrum where
    P(0)=0, or sqrt(k1^2+k2^2) at the origin) autodiff produces 0*inf = NaN.
    Uses the double-where trick: evaluate sqrt on a safe positive value
    whenever x <= 0, and select the true branch with jnp.where.
    """
    positive = x > 0
    x_safe = jnp.where(positive, x, 1.0)
    return jnp.where(positive, jnp.sqrt(x_safe), 0.0)


# ---------------------------------------------------------------------------
# Smooth clamp (replaces hard set_nan_and_neg_inf_to_min_plus_offset)
# ---------------------------------------------------------------------------


def soft_clamp_low(x, floor, scale=0.01):
    """Smooth lower clamp: approaches `floor` from above, equals x when x >> floor.

    Uses softplus: result = floor + scale * softplus((x - floor) / scale).
    For x >> floor + scale: result ≈ x (identity).
    For x << floor: result ≈ floor.
    The transition width is controlled by `scale` (smaller = sharper).
    """
    return floor + jax.nn.softplus((x - floor) / scale) * scale


# ---------------------------------------------------------------------------
# Smooth max (replaces jnp.max for normalization shifts)
# ---------------------------------------------------------------------------


def smooth_max(x, axis=None, sharpness=1000.0):
    """Differentiable approximation to max via scaled logsumexp.

    smooth_max(x) = logsumexp(sharpness * x) / sharpness
    As sharpness -> inf, this approaches the true max.
    Default sharpness=1000 gives sub-0.1% error for typical use cases.
    """
    return jax.scipy.special.logsumexp(sharpness * x, axis=axis) / sharpness


def logistic(x, x0, k):
    # function is one above x0 (for positive k)
    # function is zero bewlo x0 (for positive k)
    # smooted out over scale 1 / k)
    return 1 / (1 + jnp.exp(-k * (x - x0)))


def histdd(x, bin_list):
    return jnp.histogramdd(x, bins=bin_list)


# Vectorize over the first argument of histdd_v2
histdd_v = jax.vmap(histdd, in_axes=(0, None))


def compute_centers_and_delta_from_array(x):
    deltas = x[1:] - x[:-1]
    centers = x[:-1] + deltas / 2
    return centers, deltas


def subtract_mean(x):
    return x - jnp.mean(x)


def subtract_max(x):
    return x - jnp.max(x)


# convolutions
# fix keyword to mode="same", otherwise vmap will not work since
# one cannot pass non-vectorized keywords to vmapped functions
convv_1d = jax.tree_util.Partial(jnp.convolve, mode="same")


def get_jnp_array_or_None(x):
    if x is None:
        return None
    return jnp.array(x)


def normalize_probs_over_axes(probs, axes=None):

    norm = jnp.sum(probs, axis=axes, keepdims=True)
    probs_conditioned = probs / norm

    return probs_conditioned


def linear_array_from_dict(d):
    return jnp.linspace(d["range"][0], d["range"][1], d["shape"])


hist_v = jax.vmap(jnp.histogramdd, in_axes=(0, None))
hist_vv = jax.vmap(jnp.histogramdd, in_axes=(0, None, None, 0))
hist_vv_with_bins = jax.vmap(jnp.histogramdd, in_axes=(0, 0, None, 0))


def check_normalization(max_dev, tol, debug=False):
    def on_fail(_):
        jax.debug.print(
            "Normalization failed: max|sum_x p*Δx - 1| = {max_dev:.3e} > tol={tol}",
            max_dev=max_dev,
            tol=tol,
        )

    if debug:
        return jax.lax.cond(max_dev > tol, on_fail, lambda _: None, operand=None)
    else:
        return True


SQRT2 = jnp.sqrt(2.0)


def smooth_heaviside_erf(x: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """
    Smooth Heaviside H(x) ≈ Φ(x / sigma) = 0.5 * (1 + erf(x / (sqrt(2)*sigma))).

    Args:
      x: input tensor
      sigma: smoothing scale (>0). Smaller = sharper step. As sigma→0, → hard step.

    Returns:
      Values in (0, 1), with H(0)=0.5. Differentiable everywhere.
    """
    sigma = jnp.asarray(sigma)
    z = x / (SQRT2 * sigma)
    return 0.5 * (1.0 + jax.scipy.special.erf(z))


def log_smooth_heaviside_erf(x: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    xx = x / sigma
    # -log(1 + exp(-xx)) via logaddexp. The direct form overflows exp() for
    # xx < -709.8 in float64 (-87.3 in float32), giving -inf and, worse, a NaN
    # derivative -- which propagates to every parameter the window depends on
    # (mass_min, mass_max, sigma_*). Grid points far outside the mass window are
    # routine, so this is reached in normal use.
    return -jnp.logaddexp(0.0, -xx)


def log_smooth_heaviside_window(
    x: jnp.ndarray,
    xmin: float,
    xmax: float,
    sigma_low: float = 1.0,
    sigma_high: float = 1.0,
) -> jnp.ndarray:
    """
    Smooth log Heaviside window between [xmin, xmax].
    Equivalent to log(H(x - xmin)) + log(1 - H(x - xmax)), computed stably.

    Args:
      x: input tensor
      xmin: lower transition center
      xmax: upper transition center
      sigma: smoothing scale

    Returns:
      log H_window(x) ∈ (-∞, 0], smooth and stable
    """
    log_left = log_smooth_heaviside_erf(x - xmin, sigma_low)
    log_right = log_smooth_heaviside_erf(-x + xmax, sigma_high)
    return log_left + log_right


def rbf_kernel_1d(x, amp, rho, jitter=1e-6):
    d2 = (x[:, None] - x[None, :]) ** 2
    K = (amp**2) * jnp.exp(-0.5 * d2 / (rho**2))
    return K + jitter * jnp.eye(x.shape[0])


def set_nan_and_neg_inf_to_min_plus_offset(x, offset=0.0, softness=1.0):
    """Replace non-finite values and smoothly clamp all values above a floor.

    First substitutes NaN/inf with zeros (stopping gradient flow through bad entries),
    then applies a smooth lower clamp so the gradient tapers continuously near the floor
    rather than jumping discontinuously.
    """
    is_bad = ~jnp.isfinite(x)
    x_safe = jnp.where(is_bad, jnp.zeros_like(x), x)
    xmin = jnp.min(x_safe)
    floor = xmin + offset
    # Smooth clamp: values well above floor pass through unchanged,
    # values near or below floor get smoothly pushed up.
    return jnp.where(is_bad, floor, soft_clamp_low(x_safe, floor, scale=softness))


def handle_nonfinite_log_probs(log_probs, analysis, label, offset=-100.0):
    """Control how non-finite likelihood terms are handled.

    By default we keep non-finite values unchanged so they remain visible to the
    sampler and diagnostics. The legacy repair path remains available via
    ``likelihood_evaluation.nonfinite_log_prob_policy: repair``.
    """
    likelihood_eval_kwargs = analysis.kwargs_analysis.get("likelihood_evaluation", {})
    policy = likelihood_eval_kwargs.get("nonfinite_log_prob_policy", "repair")

    if policy == "repair":
        return set_nan_and_neg_inf_to_min_plus_offset(log_probs, offset=offset)

    if policy != "strict":
        raise ValueError(
            f"Unknown nonfinite_log_prob_policy='{policy}'. "
            "Choose from: 'strict', 'repair'."
        )

    nonfinite_count = jnp.sum(~jnp.isfinite(log_probs))

    def _print_nonfinite(_):
        jax.debug.print(
            "{label}: encountered {count} non-finite log-prob entries",
            label=label,
            count=nonfinite_count,
        )

    jax.lax.cond(nonfinite_count > 0, _print_nonfinite, lambda _: None, operand=None)
    return log_probs


# these functions fix the axis, since otherwise the max would be applied to the wrong
# axis in general
def logsumexp_last_axis(x):
    axis = -1
    # For numerical stability, subtract the maximum value before exponentiating
    x_max = jnp.max(x, axis=axis, keepdims=False)
    return jnp.log(jnp.sum(jnp.exp(x - x_max[..., None]), axis=axis)) + x_max


def logmeanexp_last_axis(x):
    axis = -1
    # For numerical stability, subtract the maximum value before exponentiating
    x_max = jnp.max(x, axis=axis, keepdims=False)
    return jnp.log(jnp.mean(jnp.exp(x - x_max[..., None]), axis=axis)) + x_max


def log_estimator_and_variance(log_weights):
    """Numerically stable log-mean and variance from 1D log-weights."""
    x_max = jnp.max(log_weights)
    w = jnp.exp(log_weights - x_max)

    mean_w = jnp.mean(w)
    log_mean = jnp.log(mean_w) + x_max

    mean_w2 = jnp.mean(w**2)
    variance = (mean_w2 - mean_w**2) / w.shape[0]
    log_variance = jnp.log(jnp.abs(variance)) + 2 * x_max

    return log_mean, log_variance


def _log_estimator_and_variance_uniform(log_weights, n):
    """Fast path for uniform group sizes: reshape to 2D and use array reductions."""
    num_groups = n.shape[0]
    group_size = log_weights.shape[0] // num_groups
    x = log_weights.reshape(num_groups, group_size)

    # per-group max for numerical stability
    x_max = jnp.max(x, axis=1, keepdims=True)
    w = jnp.exp(x - x_max)

    mean_w = jnp.mean(w, axis=1)
    log_means = jnp.log(mean_w) + x_max[:, 0]

    # Var(mean_estimator) = (E[w^2] - E[w]^2) / n
    mean_w2 = jnp.mean(w**2, axis=1)
    variances = (mean_w2 - mean_w**2) / group_size
    log_variances = jnp.log(jnp.abs(variances)) + 2 * x_max[:, 0]

    return log_means, log_variances


def make_segment_ids_from_group_sizes(n):
    """Create 1D segment IDs for contiguously stacked groups."""
    total = int(jnp.sum(jnp.asarray(n)))
    return jnp.repeat(
        jnp.arange(len(n), dtype=jnp.int32),
        jnp.asarray(n, dtype=jnp.int32),
        total_repeat_length=total,
    )


def group_sizes_are_uniform(n):
    """Return True when all group sizes are identical."""
    n = jnp.asarray(n)
    return bool(jnp.all(n == n[0]))


def _log_estimator_and_variance_variable(log_weights, n, segment_ids=None):
    """Fallback for variable group sizes using segment_sum."""
    num_groups = n.shape[0]
    if segment_ids is None:
        total = log_weights.shape[0]
        segment_ids = jnp.repeat(jnp.arange(num_groups), n, total_repeat_length=total)

    # Per-group max, matching the uniform path. A single global max would force
    # every group to share one exponent, so any event whose log-weights sit more
    # than ~708 nats (float64) or ~87 nats (float32) below the best event in the
    # whole catalog underflows to w = 0, giving log_mean = -inf and a NaN gradient.
    log_w_max = jax.ops.segment_max(log_weights, segment_ids, num_segments=num_groups)
    w = jnp.exp(log_weights - log_w_max[segment_ids])

    sums = jax.ops.segment_sum(w, segment_ids, num_segments=num_groups)
    sums_sq = jax.ops.segment_sum(w**2, segment_ids, num_segments=num_groups)

    means = sums / n
    log_means = jnp.log(means) + log_w_max

    variances = sums_sq / n**2 - means**2 / n
    log_variances = jnp.log(jnp.abs(variances)) + 2 * log_w_max

    return log_means, log_variances


def log_estimator_and_variance_stacked(
    log_weights, n, uniform_groups=True, segment_ids=None
):
    """Numerically stable log-mean and variance per group from stacked 1D log-weights.

    Args:
        log_weights: 1D array of log-weights, groups concatenated contiguously.
        n: 1D array of group sizes (may vary per event).
        uniform_groups: if True (default), uses a fast reshape path assuming
            all groups have the same size. Set to False for variable group sizes.
        segment_ids: optional precomputed 1D segment IDs. If supplied, the
            variable group-size path is used without constructing segment IDs
            inside compiled code.

    Returns:
        log_means: log of per-group mean of exp(log_weights).
        log_variances: log of per-group variance of the mean estimator.
    """
    if segment_ids is not None:
        return _log_estimator_and_variance_variable(
            log_weights, n, segment_ids=segment_ids
        )
    if uniform_groups:
        return _log_estimator_and_variance_uniform(log_weights, n)
    return _log_estimator_and_variance_variable(log_weights, n)


def logmeanexp(x, counts=None, axis=None):
    max_l = jnp.max(x, axis=axis, keepdims=True)
    prob = jnp.exp(x - max_l)
    if counts is not None:
        prob *= counts
    return jnp.log(jnp.mean(prob, axis=axis)) + max_l.squeeze(axis)


def sum_log_probs(log_probs_dict, batch_ndims=1):
    """
    Sums log probabilities in a dictionary, preserving the first `batch_ndims`
    and summing over all remaining 'event' dimensions.
    """

    def sum_event_dims(x):
        # We want to sum over axes starting from batch_ndims up to the end.
        # If x has fewer dimensions than batch_ndims, the range is empty
        # and sum(axis=()) returns x unchanged (correct behavior).
        reduction_axes = tuple(range(batch_ndims, x.ndim))
        return x.sum(axis=reduction_axes)

    # 1. Collapse event dims for each variable individually
    reduced_dict = tree_map(sum_event_dims, log_probs_dict)

    # 2. Sum all variables together
    total_log_prob = tree_reduce(operator.add, reduced_dict)

    return total_log_prob


def get_relative_variance(log_mean, log_var):
    """
    Compute log(relative variance) = log(Var(estimator) / mean(estimator)^2) = log(Var(estimator)) - 2*log(mean(estimator))

    """

    log_relative_variance = log_var - 2 * log_mean
    return log_relative_variance


def get_penalty_factor_relative_variance(
    log_relative_variance_E, log_relative_variance_selection, strength
):

    num_events = log_relative_variance_E.shape[-1]

    relative_variance_E = jnp.exp(log_relative_variance_E)
    relative_variance_selection_one = jnp.exp(log_relative_variance_selection)
    relative_variance_selection = relative_variance_selection_one * num_events**2

    relative_variance = (
        jnp.sum(relative_variance_E, axis=-1) + relative_variance_selection
    )
    factor = (relative_variance - 1) ** 2

    return -strength * factor * smooth_heaviside_erf(relative_variance - 1, sigma=0.01)
