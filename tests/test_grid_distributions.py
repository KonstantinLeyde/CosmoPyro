import jax
import jax.numpy as jnp
import numpy as np

from cosmopyro.distributions.grid_distributions import (
    InterpolatedConditional1D,
    SeparableConditional1D,
    normalize_cond_interpolated_1d,
    piecewise_linear_quadrature_weights,
)

jax.config.update("jax_enable_x64", True)

EPSILON = 1e-12


def _ref_locate(centers, x):
    """searchsorted-based bracketing, as in the pre-fast-path implementation."""
    i0 = np.clip(np.searchsorted(centers, x, side="right") - 1, 0, len(centers) - 2)
    c0, c1 = centers[i0], centers[i0 + 1]
    t = np.clip((x - c0) / (c1 - c0), 0.0, 1.0)
    return i0, t


def _ref_log_prob(x_edges, cond, x, y_edges=None, y=None, continuous_y=False):
    """NumPy reference for linear_density log_prob with at most one y variable."""
    x_edges = np.asarray(x_edges)
    cond = np.asarray(cond)
    centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    oob = (x < x_edges[0]) | (x >= x_edges[-1])
    i0, t = _ref_locate(centers, x)

    if y_edges is None:
        dens = (1 - t) * cond[i0] + t * cond[i0 + 1]
    else:
        y_edges = np.asarray(y_edges)
        oob = oob | (y < y_edges[0]) | (y >= y_edges[-1])
        if continuous_y:
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            j0, s = _ref_locate(y_centers, y)
            dens = (
                (1 - t) * (1 - s) * cond[i0, j0]
                + (1 - t) * s * cond[i0, j0 + 1]
                + t * (1 - s) * cond[i0 + 1, j0]
                + t * s * cond[i0 + 1, j0 + 1]
            )
        else:
            j = np.clip(
                np.searchsorted(y_edges, y, side="right") - 1, 0, len(y_edges) - 2
            )
            dens = (1 - t) * cond[i0, j] + t * cond[i0 + 1, j]

    return np.log(np.where(oob, EPSILON, dens))


def _ref_log_mass(x_edges, cond, x):
    x_edges = np.asarray(x_edges)
    widths = np.diff(x_edges)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, len(widths) - 1)
    log_dens = _ref_log_prob(x_edges, cond, x)
    oob = (x < x_edges[0]) | (x >= x_edges[-1])
    return np.where(oob, np.log(EPSILON), log_dens + np.log(widths[x_idx]))


def _draws(rng, lo, hi, n):
    """Draws inside the range plus some out-of-bounds on either side."""
    span = hi - lo
    return np.concatenate(
        [
            rng.uniform(lo, hi, n),
            rng.uniform(lo - span, lo, n // 4),
            rng.uniform(hi, hi + span, n // 4),
        ]
    )


def test_uniform_and_nonuniform_x_match_reference():
    rng = np.random.default_rng(0)
    edges_by_case = {
        "uniform": np.linspace(0.0, 3.0, 41),
        "nonuniform": np.concatenate([[0.0], np.cumsum(rng.uniform(0.02, 0.2, 40))]),
    }
    for label, edges in edges_by_case.items():
        cond = rng.uniform(0.1, 2.0, len(edges) - 1)
        cond = np.asarray(
            normalize_cond_interpolated_1d(jnp.asarray(edges), jnp.asarray(cond))
        )
        model = InterpolatedConditional1D(
            x_bins={"x": jnp.asarray(edges)}, y_bins=None, cond=jnp.asarray(cond)
        )
        x = _draws(rng, edges[0], edges[-1], 500)

        got = np.asarray(model.log_prob({"x": jnp.asarray(x)}))
        want = _ref_log_prob(edges, cond, x)
        np.testing.assert_allclose(got, want, rtol=1e-10, err_msg=label)

        got_mass = np.asarray(model.log_mass({"x": jnp.asarray(x)}))
        want_mass = _ref_log_mass(edges, cond, x)
        np.testing.assert_allclose(got_mass, want_mass, rtol=1e-10, err_msg=label)


def test_discrete_y_matches_reference():
    rng = np.random.default_rng(1)
    x_edges = np.linspace(0.0, 2.0, 31)
    n_y = 12
    y_edges = np.arange(n_y + 1, dtype=float)
    cond = rng.uniform(0.1, 2.0, (30, n_y))
    model = InterpolatedConditional1D(
        x_bins={"x": jnp.asarray(x_edges)},
        y_bins={"y": jnp.asarray(y_edges)},
        cond=jnp.asarray(cond),
    )
    x = _draws(rng, x_edges[0], x_edges[-1], 400)
    y = rng.integers(-2, n_y + 2, len(x)).astype(float)

    got = np.asarray(model.log_prob({"x": jnp.asarray(x)}, {"y": jnp.asarray(y)}))
    want = _ref_log_prob(x_edges, cond, x, y_edges=y_edges, y=y)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_continuous_y_matches_reference():
    rng = np.random.default_rng(2)
    x_edges = np.linspace(0.0, 2.0, 21)
    y_edges = np.concatenate([[0.0], np.cumsum(rng.uniform(0.05, 0.3, 15))])
    cond = rng.uniform(0.1, 2.0, (20, 15))
    model = InterpolatedConditional1D(
        x_bins={"x": jnp.asarray(x_edges)},
        y_bins={"y": jnp.asarray(y_edges)},
        cond=jnp.asarray(cond),
        continuous_y_names=["y"],
    )
    x = _draws(rng, x_edges[0], x_edges[-1], 400)
    y = _draws(rng, y_edges[0], y_edges[-1], 400)[: len(x)]

    got = np.asarray(model.log_prob({"x": jnp.asarray(x)}, {"y": jnp.asarray(y)}))
    want = _ref_log_prob(x_edges, cond, x, y_edges=y_edges, y=y, continuous_y=True)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_quadrature_weights_match_normalization():
    rng = np.random.default_rng(3)
    edges = np.concatenate([[0.0], np.cumsum(rng.uniform(0.02, 0.2, 25))])
    cond = rng.uniform(0.1, 2.0, 25)
    q = piecewise_linear_quadrature_weights(jnp.asarray(edges))
    cond_norm = normalize_cond_interpolated_1d(jnp.asarray(edges), jnp.asarray(cond))
    np.testing.assert_allclose(float(q @ cond_norm), 1.0, rtol=1e-12)


def test_separable_matches_joint_at_centers():
    rng = np.random.default_rng(4)
    x_edges = np.linspace(0.0, 1.5, 26)
    n_y = 12
    y_edges = np.arange(n_y + 1, dtype=float)
    prob_x = np.asarray(
        normalize_cond_interpolated_1d(
            jnp.asarray(x_edges), jnp.asarray(rng.uniform(0.1, 2.0, 25))
        )
    )
    sky = rng.uniform(0.1, 3.0, (25, n_y))

    joint = normalize_cond_interpolated_1d(
        jnp.asarray(x_edges), jnp.asarray(prob_x[:, None] * sky)
    )
    model_joint = InterpolatedConditional1D(
        x_bins={"x": jnp.asarray(x_edges)},
        y_bins={"y": jnp.asarray(y_edges)},
        cond=joint,
    )
    model_sep = SeparableConditional1D(
        x_bins={"x": jnp.asarray(x_edges)},
        y_bins={"y": jnp.asarray(y_edges)},
        prob_x=jnp.asarray(prob_x),
        grid=jnp.asarray(sky),
    )

    centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    xg, yg = np.meshgrid(centers, np.arange(n_y, dtype=float), indexing="ij")
    lp_joint = np.asarray(
        model_joint.log_prob({"x": jnp.asarray(xg)}, {"y": jnp.asarray(yg)})
    )
    lp_sep = np.asarray(
        model_sep.log_prob({"x": jnp.asarray(xg)}, {"y": jnp.asarray(yg)})
    )
    np.testing.assert_allclose(lp_sep, lp_joint, rtol=1e-10)

    # off-center values differ only at interpolation order; both stay finite
    rng_x = rng.uniform(x_edges[0], x_edges[-1], 200)
    rng_y = rng.integers(0, n_y, 200).astype(float)
    lp = np.asarray(
        model_sep.log_prob({"x": jnp.asarray(rng_x)}, {"y": jnp.asarray(rng_y)})
    )
    assert np.all(np.isfinite(lp))

    # out-of-bounds in x or y gives exactly log(epsilon)
    lp_oob = np.asarray(
        model_sep.log_prob(
            {"x": jnp.asarray([-1.0, 0.5, 0.5])},
            {"y": jnp.asarray([0.0, -1.0, n_y + 1.0])},
        )
    )
    np.testing.assert_allclose(lp_oob, np.log(EPSILON))


def test_separable_uniform_sky_matches_marginal():
    rng = np.random.default_rng(5)
    x_edges = np.linspace(0.0, 1.0, 21)
    n_y = 8
    y_edges = np.arange(n_y + 1, dtype=float)
    prob_x = np.asarray(
        normalize_cond_interpolated_1d(
            jnp.asarray(x_edges), jnp.asarray(rng.uniform(0.1, 2.0, 20))
        )
    )

    model_marginal = InterpolatedConditional1D(
        x_bins={"x": jnp.asarray(x_edges)}, y_bins=None, cond=jnp.asarray(prob_x)
    )
    model_sep = SeparableConditional1D(
        x_bins={"x": jnp.asarray(x_edges)},
        y_bins={"y": jnp.asarray(y_edges)},
        prob_x=jnp.asarray(prob_x),
        grid=None,
    )

    x = rng.uniform(x_edges[0], x_edges[-1], 300)
    y = rng.integers(0, n_y, 300).astype(float)
    lp_sep = np.asarray(
        model_sep.log_prob({"x": jnp.asarray(x)}, {"y": jnp.asarray(y)})
    )
    lp_marg = np.asarray(model_marginal.log_prob({"x": jnp.asarray(x)}))
    np.testing.assert_allclose(lp_sep, lp_marg, rtol=1e-12)


def test_separable_log_prob_has_finite_gradient():
    x_edges = jnp.linspace(0.0, 1.0, 21)
    y_edges = jnp.arange(5.0)
    sky = jnp.ones((20, 4)).at[3, 2].set(5.0)

    def lp(scale):
        prob_x = normalize_cond_interpolated_1d(
            x_edges, jnp.exp(-scale * jnp.linspace(0.0, 1.0, 20))
        )
        model = SeparableConditional1D(
            x_bins={"x": x_edges},
            y_bins={"y": y_edges},
            prob_x=prob_x,
            grid=sky,
        )
        return jnp.sum(
            model.log_prob(
                {"x": jnp.array([0.31, 0.77, -1.0])},
                {"y": jnp.array([2.0, 1.0, 0.0])},
            )
        )

    grad = jax.grad(lp)(1.3)
    assert jnp.isfinite(grad)


def test_smooth_log_interpolation_has_continuous_x_gradient_at_center():
    edges = jnp.linspace(0.0, 4.0, 5)
    cond = jnp.exp(jnp.array([0.0, 1.0, -0.5, 0.2]))
    model = InterpolatedConditional1D(
        x_bins={"x": edges},
        y_bins=None,
        cond=cond,
        interpolation="smooth_log",
    )

    def log_prob_at(x):
        return model.log_prob({"x": x})

    center = 1.5
    eps = 1e-8
    grad_left = jax.grad(log_prob_at)(center - eps)
    grad_right = jax.grad(log_prob_at)(center + eps)

    assert jnp.all(jnp.isfinite(jnp.array([grad_left, grad_right])))
    assert jnp.isclose(grad_left, grad_right, atol=1e-6)


def test_smooth_log_interpolation_has_continuous_y_gradient_at_center():
    x_edges = jnp.linspace(0.0, 4.0, 5)
    y_edges = jnp.linspace(0.0, 3.0, 4)
    cond = jnp.exp(
        jnp.array(
            [
                [0.0, 1.0, -0.5],
                [0.2, -0.3, 0.8],
                [0.9, 0.4, -0.1],
                [-0.4, 0.7, 0.1],
            ]
        )
    )
    model = InterpolatedConditional1D(
        x_bins={"x": x_edges},
        y_bins={"y": y_edges},
        cond=cond,
        continuous_y_names=["y"],
        interpolation="smooth_log",
    )

    def log_prob_at(y):
        return model.log_prob({"x": jnp.array(1.3)}, {"y": y})

    center = 1.5
    eps = 1e-8
    grad_left = jax.grad(log_prob_at)(center - eps)
    grad_right = jax.grad(log_prob_at)(center + eps)

    assert jnp.all(jnp.isfinite(jnp.array([grad_left, grad_right])))
    assert jnp.isclose(grad_left, grad_right, atol=1e-6)


def test_linear_density_interpolation_remains_available():
    edges = jnp.linspace(0.0, 4.0, 5)
    cond = jnp.exp(jnp.array([0.0, 1.0, -0.5, 0.2]))
    model = InterpolatedConditional1D(
        x_bins={"x": edges},
        y_bins=None,
        cond=cond,
        interpolation="linear_density",
    )

    log_prob = model.log_prob({"x": jnp.array(1.3)})

    assert jnp.isfinite(log_prob)
