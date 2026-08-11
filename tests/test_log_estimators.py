import jax
import jax.numpy as jnp

from cosmopyro.utils.jax_utils import (
    _log_estimator_and_variance_variable,
    make_segment_ids_from_group_sizes,
)


def _manual_log_estimator_and_variance(groups):
    log_means = []
    log_variances = []
    for log_weights in groups:
        weights = jnp.exp(log_weights)
        mean = jnp.mean(weights)
        variance = (jnp.mean(weights**2) - mean**2) / log_weights.shape[0]
        log_means.append(jnp.log(mean))
        log_variances.append(jnp.log(jnp.abs(variance)))
    return jnp.stack(log_means), jnp.stack(log_variances)


def test_log_estimator_and_variance_variable_two_events_five_samples_each():
    groups = jnp.array(
        [
            [-3.0, -1.0, -0.2, 0.4, 1.1],
            [-5.0, -4.0, -2.5, -0.7, 0.0],
        ]
    )
    log_weights = groups.reshape(-1)
    n = jnp.array([5, 5])

    expected_log_means, expected_log_variances = _manual_log_estimator_and_variance(
        groups
    )

    log_means, log_variances = _log_estimator_and_variance_variable(log_weights, n)

    assert jnp.allclose(log_means, expected_log_means, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        log_variances,
        expected_log_variances,
        rtol=1e-12,
        atol=1e-12,
    )


def test_log_estimator_and_variance_variable_accepts_precomputed_segment_ids():
    groups = jnp.array(
        [
            [-3.0, -1.0, -0.2, 0.4, 1.1],
            [-5.0, -4.0, -2.5, -0.7, 0.0],
        ]
    )
    log_weights = groups.reshape(-1)
    n = jnp.array([5, 5])
    segment_ids = make_segment_ids_from_group_sizes(n)

    expected_log_means, expected_log_variances = _manual_log_estimator_and_variance(
        groups
    )

    log_means, log_variances = _log_estimator_and_variance_variable(
        log_weights,
        n,
        segment_ids=segment_ids,
    )

    assert jnp.allclose(log_means, expected_log_means, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        log_variances,
        expected_log_variances,
        rtol=1e-12,
        atol=1e-12,
    )


def test_log_estimator_and_variance_variable_uses_per_group_max():
    """One event far below the catalog best must not underflow the others.

    With a single global max, every group shares one exponent, so a group sitting
    more than ~708 nats (float64) below the best group underflows to zero weight,
    giving log_mean = -inf and a NaN gradient. A per-group max keeps the dynamic
    range budget per event.
    """
    offset = -5000.0
    ns = [4, 6, 5]
    segment_ids = make_segment_ids_from_group_sizes(jnp.array(ns))
    n = jnp.array(ns)

    def log_means(theta):
        groups = [
            -((jnp.arange(size, dtype=jnp.float64) - theta) ** 2) + off
            for size, off in zip(ns, [0.0, offset, 0.0])
        ]
        stacked = jnp.concatenate(groups)
        return _log_estimator_and_variance_variable(
            stacked, n, segment_ids=segment_ids
        )[0]

    def log_means_no_offset(theta):
        groups = [-((jnp.arange(size, dtype=jnp.float64) - theta) ** 2) for size in ns]
        return _log_estimator_and_variance_variable(
            jnp.concatenate(groups), n, segment_ids=segment_ids
        )[0]

    theta = 0.3
    values = log_means(theta)
    grads = jax.jacobian(log_means)(theta)

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(grads))

    # offsetting one event shifts only that event's log_mean, by exactly `offset`
    expected = log_means_no_offset(theta).at[1].add(offset)
    assert jnp.allclose(values, expected, rtol=1e-12, atol=1e-12)

    # and no event's gradient depends on how far event 1 sits below the others
    assert jnp.allclose(
        grads, jax.jacobian(log_means_no_offset)(theta), rtol=1e-12, atol=1e-12
    )
