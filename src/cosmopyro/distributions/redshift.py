import jax.numpy as jnp

from ..distributions.grid_distributions import (
    InterpolatedConditional1D,
    normalize_cond_interpolated_1d,
)
from ..utils.jax_utils import compute_centers_and_delta_from_array

__all__ = [
    "MadauDickinsonRedshiftModel",
    "get_redshift_model",
    "madau_dickinson_2014",
]

# from Eq. 15 of https://arxiv.org/pdf/1403.0007
# GAMMA = 2.7
# ZP = 2.9
# KAPPA = 5.6


def madau_dickinson_2014(z, gamma, kappa, zp):
    """
    Computes the star formation rate density (SFRD) using the Madau & Dickinson (2014) model.

    Parameters:
    z (float or array): Redshift.
    gamma (float): Parameter for the model.
    kappa (float): Parameter for the model.
    z_p (float): Pivot redshift.

    Returns:
    float or array: SFRD in units of solar masses per year per cubic megaparsec.
    """

    # taken from the reviewed icarogw code
    # https://github.com/simone-mastrogiovanni/icarogw/blob/main/icarogw/cosmology.py#L588
    log_prob = gamma * jnp.log1p(z) - jnp.log1p(
        jnp.power((1 + z) / (1 + zp), gamma + kappa)
    )
    return jnp.exp(log_prob)


def get_redshift_model(
    name,
    cosmological_model,
    parameters,
    redshift_boundaries=None,
    interpolation="smooth_log",
):
    if name == "MadauDickinson":
        return MadauDickinsonRedshiftModel(
            cosmological_model,
            parameters,
            redshift_boundaries=redshift_boundaries,
            interpolation=interpolation,
        )
    else:
        raise ValueError(
            f"Unknown redshift model: {name}. Available models: ['MadauDickinson']"
        )


class MadauDickinsonRedshiftModel(InterpolatedConditional1D):
    def __init__(
        self,
        cosmological_model,
        parameters,
        redshift_boundaries=None,
        interpolation="smooth_log",
    ):
        self.cosmological_model = cosmological_model
        self.parameters = parameters

        if redshift_boundaries is None:
            self.z_boundaries = self.cosmological_model.z_interp
        else:
            self.z_boundaries = redshift_boundaries

        x_bins = dict(redshift=self.z_boundaries)

        self.initialize_redshift_distribution()

        super().__init__(
            x_bins=x_bins,
            y_bins=None,
            cond=self.p_z_interp,
            interpolation=interpolation,
        )

    def initialize_redshift_distribution(self):

        # need the probability on centers, not boundaries
        z, _ = compute_centers_and_delta_from_array(self.z_boundaries)
        params = self.parameters.get("redshift", {})

        p_z_MD = madau_dickinson_2014(z, params["gamma"], params["kappa"], params["zp"])
        p_z_comoving = self.cosmological_model.get_comoving_volume_differential(z)
        p_z_time_dilation = 1 / (1 + z)

        p_z_nn = p_z_MD * p_z_comoving * p_z_time_dilation

        self.p_z_interp = normalize_cond_interpolated_1d(
            x_edges=self.z_boundaries, cond=p_z_nn
        )
