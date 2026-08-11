import jax.numpy as jnp
import numpy as np

__all__ = [
    "add_m1m2_to_dict",
    "get_chirp_mass",
    "get_component_masses_from_chirp_mass_and_mass_ratio",
    "get_jacobian_chirp_mass_mass_ratio_from_component_masses",
    "get_log_jacobian_chirp_mass_d_dL_to_mass_1_s_redshift",
    "get_mass_1_from_chirp_mass_and_mass_ratio",
    "reweight_samples",
]


def get_chirp_mass(m1, m2):
    m1 = abs(m1)
    m2 = abs(m2)
    return m1 ** (3 / 5) * m2 ** (3 / 5) / (m1 + m2) ** (1 / 5)


def get_component_masses_from_chirp_mass_and_mass_ratio(chirp_mass, mass_ratio):

    m1 = (
        chirp_mass * jnp.power(1 + mass_ratio, 1.0 / 5) / jnp.power(mass_ratio, 3.0 / 5)
    )
    m2 = m1 * mass_ratio
    return m1, m2


def get_mass_1_from_chirp_mass_and_mass_ratio(chirp_mass, mass_ratio):
    return chirp_mass * (1 + mass_ratio) ** (1 / 5) / mass_ratio ** (3 / 5)


def get_jacobian_chirp_mass_mass_ratio_from_component_masses(m1, m2):

    chirp_mass = get_chirp_mass(m1, m2)

    return chirp_mass / m1**2


def add_m1m2_to_dict(dict_in):

    dict_in["mass_1"], dict_in["mass_2"] = (
        get_component_masses_from_chirp_mass_and_mass_ratio(
            dict_in["chirp_mass"], dict_in["mass_ratio"]
        )
    )


def reweight_samples(samples, weights, N_samples=None):
    """
    The samples are reweighted as a function of the given weights.

    """

    if N_samples == None:
        N_samples = len(samples)

    if np.any(np.isinf(weights)):
        print("Infinite weights encountered. ")

    # replace infinite weights
    weights = np.where(np.isinf(weights), 0, weights)
    weights = np.where(np.isnan(weights), 0, weights)

    if np.any(np.all(weights == 0)):
        print("***")
        print("Warning: Weights are all zero. ")
        print("***")

    idx = np.random.choice(len(weights), size=N_samples, p=weights / np.sum(weights))

    return samples[idx], idx


def get_log_jacobian_chirp_mass_d_dL_to_mass_1_s_redshift(
    cosmological_model, redshift, mass_1_s, mass_ratio
):
    return (
        jnp.log(1 + redshift)
        + jnp.log(2 + 3 * mass_ratio)
        - jnp.log(5)
        - jnp.log(mass_1_s)
        - jnp.log(1 + mass_ratio)
        + jnp.log(
            cosmological_model.get_dluminosity_distance_gw_over_dz_from_redshift(
                redshift
            )
        )
    )
