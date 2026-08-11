import equinox as eqx
import jax.numpy as jnp

from ..utils.jax_utils import compute_centers_and_delta_from_array, smooth_interp

__all__ = [
    "C_KM_PER_SEC",
    "Z_MAX",
    "Z_MIN",
    "Z_STEPS",
    "FlatLambdaCDM",
    "interp_z_in_log_distance_over_H0",
    "interp_z_in_log_distance_over_H0_inverse",
]

C_KM_PER_SEC = 299_792.458  # km/s

Z_MIN = 1e-5
Z_MAX = 3
Z_STEPS = 200_000


def _interp_dispatch(x_query, x_knots, y_knots, method="smooth", **kwargs):
    """Dispatch to smooth_interp (cubic Hermite) or jnp.interp (linear)."""
    if method == "smooth":
        return smooth_interp(x_query, x_knots, y_knots, **kwargs)
    return jnp.interp(x_query, x_knots, y_knots, **kwargs)


def interp_z_in_log_distance_over_H0(
    x, log_x_interp, log_y_interp, H0, method="smooth", **kwargs
):
    log_x = jnp.log(x)
    log_y = _interp_dispatch(log_x, log_x_interp, log_y_interp, method=method, **kwargs)
    log_y = log_y - jnp.log(H0)
    return jnp.exp(log_y)


def interp_z_in_log_distance_over_H0_inverse(
    y, log_x_interp, log_y_interp, H0, method="smooth", **kwargs
):
    log_y = jnp.log(y) + jnp.log(H0)
    log_x = _interp_dispatch(log_y, log_y_interp, log_x_interp, method=method, **kwargs)
    return jnp.exp(log_x)


def _initialize_comoving_distance(Omega_m, numerics):
    """Precompute interpolation tables (independent of H0)."""
    z_interp = jnp.logspace(
        jnp.log10(numerics["z_min"]),
        jnp.log10(numerics["z_max"]),
        numerics["z_steps"],
    )
    _, delta_z = compute_centers_and_delta_from_array(z_interp)

    limit_value_left = C_KM_PER_SEC * numerics["z_min"]

    Omega_Lambda = 1.0 - Omega_m
    integrand = C_KM_PER_SEC / jnp.sqrt(Omega_m * (1 + z_interp) ** 3 + Omega_Lambda)

    incs = 0.5 * delta_z * (integrand[:-1] + integrand[1:])
    integral_cum = (
        jnp.concatenate([jnp.array([0], dtype=integrand.dtype), jnp.cumsum(incs)])
        + limit_value_left
    )

    log_z_interp = jnp.log(z_interp)
    log_comoving = jnp.log(integral_cum)
    log_luminosity = log_comoving + jnp.log(1 + z_interp)
    log_angular = log_comoving - jnp.log(1 + z_interp)

    return z_interp, log_z_interp, log_comoving, log_luminosity, log_angular


class FlatLambdaCDM(eqx.Module):
    """Flat LCDM cosmology with H0 as a JAX-traced attribute."""

    # JAX-traced leaves
    H0: float
    _Omega_m: float

    # Precomputed interpolation tables (traced but depend only on Omega_m)
    z_interp: jnp.ndarray
    log_z_interp: jnp.ndarray
    log_comoving_distance_interp_wo_H0: jnp.ndarray
    log_luminosity_distance_interp_wo_H0: jnp.ndarray
    log_angular_diameter_distance_interp_wo_H0: jnp.ndarray

    # Params dict (contains H0 which is JAX-traced)
    params: dict

    # Static metadata (not JAX-traced)
    numerics: dict = eqx.field(static=True)
    interpolation_kwargs: dict = eqx.field(static=True)

    def __init__(self, params, numerics=None):
        if numerics is None:
            numerics = dict(z_min=Z_MIN, z_max=Z_MAX, z_steps=Z_STEPS)

        self.H0 = params.get("H0", 70.0)
        self.params = params
        self._Omega_m = params["Omega_m"]
        self.numerics = numerics
        self.interpolation_kwargs = {"left": "extrapolate", "right": None}

        (
            self.z_interp,
            self.log_z_interp,
            self.log_comoving_distance_interp_wo_H0,
            self.log_luminosity_distance_interp_wo_H0,
            self.log_angular_diameter_distance_interp_wo_H0,
        ) = _initialize_comoving_distance(self._Omega_m, numerics)

    def get_cosmological_parameters(self):
        return self.params

    def get_luminosity_distance_from_redshift(self, z):
        return interp_z_in_log_distance_over_H0(
            z,
            self.log_z_interp,
            self.log_luminosity_distance_interp_wo_H0,
            H0=self.H0,
            **self.interpolation_kwargs,
        )

    def get_angular_diameter_distance_from_z(self, z):
        return interp_z_in_log_distance_over_H0(
            z,
            self.log_z_interp,
            self.log_angular_diameter_distance_interp_wo_H0,
            H0=self.H0,
            **self.interpolation_kwargs,
        )

    def get_comoving_distance_from_redshift(self, z):
        return interp_z_in_log_distance_over_H0(
            z,
            self.log_z_interp,
            self.log_comoving_distance_interp_wo_H0,
            H0=self.H0,
            **self.interpolation_kwargs,
        )

    def get_z_from_comoving_distance(self, comoving_distance):
        return interp_z_in_log_distance_over_H0_inverse(
            comoving_distance,
            self.log_z_interp,
            self.log_comoving_distance_interp_wo_H0,
            H0=self.H0,
            **self.interpolation_kwargs,
        )

    def get_redshift_from_luminosity_distance(self, luminosity_distance):
        return interp_z_in_log_distance_over_H0_inverse(
            luminosity_distance,
            self.log_z_interp,
            self.log_luminosity_distance_interp_wo_H0,
            H0=self.H0,
            **self.interpolation_kwargs,
        )

    def get_comoving_volume_differential(self, z):
        comoving_slice = 4 * jnp.pi * self.get_comoving_distance_from_redshift(z=z) ** 2
        return comoving_slice * self.get_dcomoving_distance_over_dz_from_redshift(z=z)

    def get_dcomoving_distance_over_dz_from_redshift(self, z):
        return self.c_over_E(z=z) / self.H0

    def get_dluminosity_distance_over_dz_from_redshift(self, z):
        comoving_distance = self.get_comoving_distance_from_redshift(z=z)
        term_1 = comoving_distance
        term_2 = (1 + z) * self.get_dcomoving_distance_over_dz_from_redshift(z=z)
        return term_1 + term_2

    def c_over_E(self, z):
        Omega_Lambda = 1.0 - self._Omega_m
        return C_KM_PER_SEC / jnp.sqrt(self._Omega_m * (1 + z) ** 3 + Omega_Lambda)

    def get_source_frame_mass(self, mass_d, redshift):
        return mass_d / (1 + redshift)

    def get_detector_frame_masses(
        self, mass_1_s, mass_2_s, luminosity_distance=None, redshift=None
    ):
        if redshift is None:
            redshift = self.get_redshift_from_luminosity_distance(
                luminosity_distance=luminosity_distance
            )
        return mass_1_s * (1 + redshift), mass_2_s * (1 + redshift)

    def get_luminosity_distance_gw_from_redshift(self, redshift):
        return self.get_luminosity_distance_from_redshift(z=redshift)

    def get_redshift_from_luminosity_distance_gw(
        self, luminosity_distance_gw, params_modified_gravity=None
    ):
        return self.get_redshift_from_luminosity_distance(
            luminosity_distance=luminosity_distance_gw
        )

    def get_dluminosity_distance_gw_over_dz_from_redshift(
        self, redshift, params_modified_gravity=None
    ):
        return self.get_dluminosity_distance_over_dz_from_redshift(z=redshift)
