# requirements: jax, matplotlib
# pip install "jax[cpu]" matplotlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

__all__ = [
    "PI",
    "draw_correlated_measurement",
    "draw_posterior_samples",
    "make_covariance",
    "plot_skymap",
    "sample_cos_prior_dec",
    "sample_uniform_ra",
    "simulate_one_event",
]

PI = jnp.pi

# ---------- Priors & Utilities ----------


def sample_cos_prior_dec(key, n):
    """Dec ~ cos(dec) on [-pi/2, pi/2] via inverse-CDF: dec = arcsin(2u-1)."""
    u = jax.random.uniform(key, shape=(n,), minval=0.0, maxval=1.0)
    return jnp.arcsin(2.0 * u - 1.0)


def sample_uniform_ra(key, n):
    """RA ~ Uniform(0, 2pi)."""
    return jax.random.uniform(key, shape=(n,), minval=0.0, maxval=2.0 * PI)


def _wrap_ra(ra):  # [0, 2pi)
    return jnp.mod(ra, 2.0 * PI)


def _clip_dec(dec):  # [-pi/2, pi/2]
    return jnp.clip(dec, -PI / 2.0, PI / 2.0)


def make_covariance(sigma_ra, sigma_dec, rho):
    """Return 2x2 covariance with correlation rho."""
    return jnp.array(
        [
            [sigma_ra**2, rho * sigma_ra * sigma_dec],
            [rho * sigma_ra * sigma_dec, sigma_dec**2],
        ]
    )


# ---------- Correlated measurement & posterior ----------


def draw_correlated_measurement(key, ra_true, dec_true, cov):
    """
    Draw a single noisy measurement from N([ra_true, dec_true], cov),
    then wrap RA and clip Dec.
    """
    mean = jnp.array([ra_true, dec_true])
    noisy = jax.random.multivariate_normal(key, mean, cov, shape=())
    ra_meas = _wrap_ra(noisy[0])
    dec_meas = _clip_dec(noisy[1])
    return ra_meas, dec_meas


def draw_posterior_samples(key, ra_center, dec_center, cov, n):
    """
    Draw n samples from N([ra_center, dec_center], cov),
    wrapping RA and clipping Dec afterward.
    """
    mean = jnp.array([ra_center, dec_center])
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(n,))
    ra_s = _wrap_ra(samples[:, 0])
    dec_s = _clip_dec(samples[:, 1])
    return jnp.stack([ra_s, dec_s], axis=1)  # (n,2) as [ra, dec]


# ---------- Plotting (Mollweide) ----------


def _ra_to_mollweide_x(ra):
    # Astronomy convention: RA increases to the left.
    return -(ra - PI)


def plot_skymap(
    ra_true, dec_true, ra_meas, dec_meas, posterior, title="Sky map (Mollweide)"
):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="mollweide")
    x_samp = _ra_to_mollweide_x(posterior["ra"])
    y_samp = posterior["dec"]
    ax.scatter(x_samp, y_samp, s=2, alpha=0.4, label="posterior samples")
    ax.scatter(
        float(_ra_to_mollweide_x(ra_meas)),
        float(dec_meas),
        s=50,
        marker="x",
        label="measurement",
    )
    ax.scatter(
        float(_ra_to_mollweide_x(ra_true)),
        float(dec_true),
        s=50,
        marker="o",
        label="truth",
    )
    ax.grid(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------- End-to-end example for one event ----------


def simulate_one_event(
    num_posterior_samples: int,
    seed: int = 0,
    meas_sigma_ra_deg: float = 5.0,
    meas_sigma_dec_deg: float = 5.0,
    rho: float = 0.2,
):
    """
    Uses the SAME correlated covariance for measurement and posterior by default.
    Returns dict with:
      'true_ra', 'true_dec', 'meas_ra', 'meas_dec' (scalars, radians)
      'posterior_samples': jnp.ndarray, shape (N,2) as [ra, dec] in radians
    """
    key = jax.random.PRNGKey(seed)
    k_ra, k_dec, k_meas, k_post = jax.random.split(key, 4)

    # Draw true position from priors
    ra_true = sample_uniform_ra(k_ra, 1)[0]
    dec_true = sample_cos_prior_dec(k_dec, 1)[0]

    # Build measurement covariance (in radians)
    meas_sigma_ra = jnp.deg2rad(meas_sigma_ra_deg)
    meas_sigma_dec = jnp.deg2rad(meas_sigma_dec_deg)
    cov_meas = make_covariance(meas_sigma_ra, meas_sigma_dec, rho)

    # Measurement from correlated Gaussian around truth
    ra_meas, dec_meas = draw_correlated_measurement(k_meas, ra_true, dec_true, cov_meas)

    # Posterior covariance — default to exactly the same as measurement
    post_sigma_ra = jnp.deg2rad(meas_sigma_ra_deg)
    post_sigma_dec = jnp.deg2rad(meas_sigma_dec_deg)
    cov_post = make_covariance(post_sigma_ra, post_sigma_dec, rho)

    # Posterior samples centered on the measurement with the same correlated covariance
    posterior = draw_posterior_samples(
        k_post, ra_meas, dec_meas, cov_post, num_posterior_samples
    )

    out = {
        "true_ra": ra_true,
        "true_dec": dec_true,
        "meas_ra": ra_meas,
        "meas_dec": dec_meas,
        "posterior_samples": {"ra": posterior[:, 0], "dec": posterior[:, 1]},
    }

    return out
