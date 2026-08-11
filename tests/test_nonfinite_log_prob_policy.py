from types import SimpleNamespace

import jax
import jax.numpy as jnp

from cosmopyro.utils.jax_utils import (
    get_penalty_factor_relative_variance,
    handle_nonfinite_log_probs,
    soft_clamp_low,
)

jax.config.update("jax_enable_x64", True)


def make_analysis(policy):
    return SimpleNamespace(
        kwargs_analysis={
            "likelihood_evaluation": {
                "nonfinite_log_prob_policy": policy,
            }
        }
    )


def test_nonfinite_log_prob_policy_strict_keeps_nonfinite_values():
    x = jnp.array([0.0, jnp.nan, -jnp.inf])
    out = handle_nonfinite_log_probs(x, make_analysis("strict"), "test")
    assert jnp.isfinite(out[0])
    assert jnp.isnan(out[1])
    assert jnp.isneginf(out[2])


def test_nonfinite_log_prob_policy_repair_replaces_nonfinite_values():
    x = jnp.array([0.0, jnp.nan, -jnp.inf])
    out = handle_nonfinite_log_probs(x, make_analysis("repair"), "test")
    assert jnp.all(jnp.isfinite(out))


def test_soft_clamp_low_has_finite_gradients():
    x = jnp.array([-1.0, -1e-6, 0.0, 1e-30, 1e-20, 1e-6, 1e-2, 1.0])

    def loss(values):
        return jnp.sum(soft_clamp_low(values, 1e-30, scale=1e-12))

    values = soft_clamp_low(x, 1e-30, scale=1e-12)
    grads = jax.grad(loss)(x)

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(grads))


def test_relative_variance_penalty_is_inactive_below_threshold():
    log_relative_variance_E = jnp.log(jnp.array([0.2, 0.3]))
    log_relative_variance_selection = jnp.log(jnp.array(1e-6))

    penalty = get_penalty_factor_relative_variance(
        log_relative_variance_E,
        log_relative_variance_selection,
        strength=1000.0,
    )

    assert jnp.isclose(penalty, 0.0, atol=1e-12)


def test_relative_variance_penalty_scales_when_active():
    log_relative_variance_E = jnp.log(jnp.array([1.5]))
    log_relative_variance_selection = jnp.log(jnp.array(1e-6))

    penalty_10 = get_penalty_factor_relative_variance(
        log_relative_variance_E,
        log_relative_variance_selection,
        strength=10.0,
    )
    penalty_100 = get_penalty_factor_relative_variance(
        log_relative_variance_E,
        log_relative_variance_selection,
        strength=100.0,
    )

    assert penalty_10 < 0.0
    assert jnp.isclose(penalty_100 / penalty_10, 10.0, rtol=1e-12)
