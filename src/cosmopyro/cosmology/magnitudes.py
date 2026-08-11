import jax.numpy as jnp

__all__ = [
    "apparent_magnitude_from_abs_magnitude",
    "compute_abs_magnitude_from_apparent_magnitude",
    "get_M_hat_from_absolute_magnitude",
    "get_absolute_magnitude_from_M_hat",
]


def compute_abs_magnitude_from_apparent_magnitude(m, luminosity_distance):
    """
    Computes the apparent magnitude from an absolute magnitude for a
    luminosity_distance in Mpc.

    Parameters
    ----------
    absolute_magnitude : float
        The absolute magnitude of the astronomical object.

    luminosity_distance : float
        The luminosity distance to the object in megaparsecs (Mpc).

    Returns
    -------
    float
        The calculated apparent magnitude.

    https://en.wikipedia.org/wiki/Luminosity_distance

    """

    return m - 5 * jnp.log10(luminosity_distance) - 25


def apparent_magnitude_from_abs_magnitude(M, luminosity_distance):
    """
    Computes the apparent magnitude from an absolute magnitude for a
    luminosity_distance in Mpc.

    Parameters
    ----------
    absolute_magnitude : float
        The absolute magnitude of the astronomical object.

    luminosity_distance : float
        The luminosity distance to the object in megaparsecs (Mpc).

    Returns
    -------
    float
        The calculated apparent magnitude.

    https://en.wikipedia.org/wiki/Luminosity_distance

    """

    return M + 5 * jnp.log10(luminosity_distance) + 25


def get_M_hat_from_absolute_magnitude(M, H0, H0_ref):
    norm_magnitudes = -5 * jnp.log10(H0 / H0_ref)
    return M + norm_magnitudes


def get_absolute_magnitude_from_M_hat(M_hat, H0, H0_ref):
    norm_magnitudes = -5 * jnp.log10(H0 / H0_ref)
    return M_hat - norm_magnitudes
