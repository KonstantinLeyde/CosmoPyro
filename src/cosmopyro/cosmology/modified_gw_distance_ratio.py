from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from ..field_utils import field
from ..utils.jax_utils import (
    compute_centers_and_delta_from_array,
    safe_sqrt,
    smooth_interp,
)
from .flat_lambdacdm import (
    Z_MAX,
    Z_MIN,
    Z_STEPS,
    FlatLambdaCDM,
    interp_z_in_log_distance_over_H0,
)

__all__ = [
    "ModifiedGWDistanceFlatLambdaCDM",
    "ModifiedGWLuminosityDistanceCosmology",
    "Omega_Lambda",
    "basis_term",
    "check_dGW_monotonicity_constraint",
    "complete_params_modified_gravity",
    "complete_ratio_params",
    "construct_ratio_gp_nodes",
    "get_luminosity_distance_GW_function_and_jacobian",
    "get_ratio_function",
    "get_ratio_function_and_jacobian",
    "get_ratio_function_jacobian",
    "integrand",
    "luminosity_distance_GW_function_gp_nodes",
    "luminosity_distance_GW_function_gp_nodes_jacobian",
    "power_spectrum_modified_gravity_gp",
    "ratio_function_cM",
    "ratio_function_cM_jacobian",
    "ratio_function_cosine",
    "ratio_function_cosine_jacobian",
    "ratio_function_gp_integrated",
    "ratio_function_gp_integrated_jacobian",
    "soft_low_clip",
    "softplus",
    "transition_func",
]


class ModifiedGWDistanceFlatLambdaCDM(FlatLambdaCDM):
    name_modified_ratio: str = eqx.field(static=True)
    ratio_fun: callable = eqx.field(static=True)
    ratio_fun_jacobian: callable = eqx.field(static=True)
    params_modified_gravity: dict

    def __init__(
        self, params, name_modified_ratio, params_modified_gravity=None, numerics=None
    ):
        if numerics is None:
            numerics = dict(z_min=Z_MIN, z_max=Z_MAX, z_steps=Z_STEPS)

        super().__init__(params, numerics=numerics)
        self.name_modified_ratio = name_modified_ratio
        self.ratio_fun, self.ratio_fun_jacobian = get_ratio_function_and_jacobian(
            name_modified_ratio
        )

        # Inject Omega_m (needed by cM parametrisation)
        if params_modified_gravity is None:
            params_modified_gravity = {}
        params_modified_gravity["Omega_m"] = self.params["Omega_m"]
        self.params_modified_gravity = params_modified_gravity

    def get_luminosity_distance_from_luminosity_distance_GW(
        self, luminosity_distance_gw
    ):
        redshift = self.get_redshift_from_luminosity_distance_gw(luminosity_distance_gw)
        return self.get_luminosity_distance_from_redshift(redshift)

    def construct_ratio(self):
        z = self.z_interp
        log_ratio_interp = jnp.log(self.ratio_fun(z, self.params_modified_gravity))
        log_luminosity_distance_gw_interp_wo_H0 = (
            log_ratio_interp + self.log_luminosity_distance_interp_wo_H0
        )
        return log_ratio_interp, log_luminosity_distance_gw_interp_wo_H0

    def get_luminosity_distance_gw_from_redshift(self, redshift):
        ratio = self.get_ratio_from_redshift(redshift)
        return ratio * self.get_luminosity_distance_from_redshift(redshift)

    def get_redshift_from_luminosity_distance_gw(
        self, luminosity_distance_gw, params_modified_gravity=None
    ):
        _, log_luminosity_distance_gw_interp_wo_H0 = self.construct_ratio()
        log_luminosity_distance_gw_interp = (
            log_luminosity_distance_gw_interp_wo_H0 - jnp.log(self.H0)
        )
        # Use linear interp for inversion: the modified d_L_gw(z) table may be
        # non-monotonic (GP-based ratios), which breaks cubic spline inversion.
        return interp_z_in_log_distance_over_H0(
            luminosity_distance_gw,
            log_luminosity_distance_gw_interp,
            self.log_z_interp,
            H0=1.0,
            method="linear",
            **self.interpolation_kwargs,
        )

    def get_dluminosity_distance_gw_over_dz_from_redshift(
        self, redshift, params_modified_gravity=None
    ):
        ratio = self.get_ratio_from_redshift(redshift)
        term1 = ratio * self.get_dluminosity_distance_over_dz_from_redshift(redshift)
        term2 = self.ratio_fun_jacobian(
            redshift, self.params_modified_gravity
        ) * self.get_luminosity_distance_from_redshift(redshift)
        return term1 + term2

    def get_ratio_from_redshift(self, redshift):
        return self.ratio_fun(redshift, self.params_modified_gravity)


def get_ratio_function_and_jacobian(name):
    if name == "FlatLambdaCDM_GW_distance_polynomial":
        # TODO
        raise NotImplementedError
    elif name == "FlatLambdaCDM_GW_distance_cosine":
        ratio_function = ratio_function_cosine
        ratio_function_jacobian = ratio_function_cosine_jacobian
    elif name == "FlatLambdaCDM_GW_distance_gp_integrated":
        ratio_function = ratio_function_gp_integrated
        ratio_function_jacobian = ratio_function_gp_integrated_jacobian
    elif name == "FlatLambdaCDM_GW_distance_cM":
        ratio_function = ratio_function_cM
        ratio_function_jacobian = ratio_function_cM_jacobian
    else:
        raise NotImplementedError

    return ratio_function, ratio_function_jacobian


def get_ratio_function(params_modified_gravity):
    # Extract the parameters for the ratio function
    A = params_modified_gravity.get("ratio_A")
    B = params_modified_gravity.get("ratio_B")
    C = params_modified_gravity.get("ratio_C")

    # Define the ratio function
    def ratio_function(z):
        return (
            1
            + A * (1 + z)
            + B * (1 + z) ** 2 / 2
            + C * (1 + z) ** 3 / 6
            - A
            - B / 2
            - C / 6
        )

    return ratio_function


def get_ratio_function_jacobian(params_modified_gravity):
    # Extract the parameters for the ratio function
    A = params_modified_gravity.get("ratio_A")
    B = params_modified_gravity.get("ratio_B")
    C = params_modified_gravity.get("ratio_C")

    # Define the Jacobian of the ratio function
    def ratio_function_jacobian(z):
        return A + B * (1 + z) + C * (1 + z) ** 2 / 2

    return ratio_function_jacobian


########################################## COS BASIS ############################################


def transition_func(z, z_tr):
    return 0.5 * (1 + erf((z - z_tr) / z_tr)) - 0.5 * (1 + erf(-1))


def basis_term(z, nu, amp, phase, z_max, z_tr):
    k = nu * jnp.pi / z_max
    # TODO: transition function can be moved outside the function
    return transition_func(z, z_tr) * amp * jnp.cos(k * z + phase)


def ratio_function_cosine(z, params_modified_gravity):

    z_max = params_modified_gravity["zmax_b"]
    z_tr = params_modified_gravity["z_tr"]

    amplitudes = params_modified_gravity["alphas"]
    phases = params_modified_gravity["phases"]

    N = amplitudes.shape[-1]

    nus = jnp.arange(1, N + 1)

    nus_1N = jnp.expand_dims(nus, -2)
    amps_1N = jnp.expand_dims(amplitudes, -2)
    phis_1N = jnp.expand_dims(phases, -2)
    z_Z1 = jnp.expand_dims(z, -1)

    f_nu = basis_term(z_Z1, nus_1N, amps_1N, phis_1N, z_max, z_tr)

    sum_of_terms = jnp.sum(f_nu, axis=-1)
    result = 1.0 + sum_of_terms

    return result.squeeze()


def ratio_function_cosine_jacobian(z, params_modified_gravity):
    """
    Builds the jacobian function using JAX's automatic differentiation.
    """
    f_grad = jax.grad(ratio_function_cosine, argnums=0)

    def fz(zz):
        return f_grad(zz, params_modified_gravity)

    return jnp.vectorize(fz)(z)


#### GP-integrated basis ########################################


def ratio_function_gp_integrated(z, params_modified_gravity):
    return smooth_interp(
        z,
        params_modified_gravity["z_nodes"],
        params_modified_gravity["ratio_nodes"],
    )


def ratio_function_gp_integrated_jacobian(z, params_modified_gravity):
    f_grad = jax.grad(ratio_function_gp_integrated, argnums=0)

    def fz(zz):
        return f_grad(zz, params_modified_gravity)

    return jnp.vectorize(fz)(z)


#### cM parametrisation #########################################


def Omega_Lambda(z, params_modified_gravity):
    Omega_m0 = params_modified_gravity["Omega_m"]
    Omega_Lambda0 = 1 - Omega_m0
    return Omega_Lambda0 / (Omega_Lambda0 + (1 + z) ** 3 * Omega_m0)


def integrand(z, params_modified_gravity):
    Omega_Lambda0 = 1 - params_modified_gravity["Omega_m"]
    return Omega_Lambda(z, params_modified_gravity) / (Omega_Lambda0 * (1 + z))


def ratio_function_cM(z, params_modified_gravity):
    from quadax import quadgk

    cM = params_modified_gravity["cM"]

    def integrate_to_zi(zi):
        def f(zz):
            return integrand(zz, params_modified_gravity)

        int_sol, _ = quadgk(f, [0, zi])
        return int_sol

    int_sol = jax.vmap(integrate_to_zi)(z)
    return jnp.exp((cM / 2.0) * int_sol)


def ratio_function_cM_jacobian(z, params_modified_gravity):
    cM = params_modified_gravity["cM"]
    ratio = ratio_function_cM(z, params_modified_gravity)
    integrand_vals = integrand(z, params_modified_gravity)
    return ratio * (cM / 2.0) * integrand_vals


class ModifiedGWLuminosityDistanceCosmology(FlatLambdaCDM):
    name_modified_ratio: str = eqx.field(static=True)
    distance_fun: callable = eqx.field(static=True)
    distance_fun_jacobian: callable = eqx.field(static=True)
    params_modified_gravity: dict

    def __init__(self, params, name, params_modified_gravity=None, numerics=None):
        if numerics is None:
            numerics = dict(z_min=1e-5, z_max=Z_MAX, z_steps=200000)

        super().__init__(params, numerics=numerics)
        self.name_modified_ratio = name
        self.distance_fun, self.distance_fun_jacobian = (
            get_luminosity_distance_GW_function_and_jacobian(name)
        )
        self.params_modified_gravity = (
            params_modified_gravity if params_modified_gravity is not None else {}
        )

    def construct_luminosity_distance_GW_interpolation(self):
        z = self.z_interp
        return jnp.log(self.distance_fun(z, self.params_modified_gravity))

    def get_luminosity_distance_from_luminosity_distance_GW(
        self, luminosity_distance_gw
    ):
        # TODO
        return None

    def get_luminosity_distance_gw_from_redshift(self, redshift):
        return self.distance_fun(redshift, self.params_modified_gravity)

    def get_redshift_from_luminosity_distance_gw(
        self, luminosity_distance_gw, params_modified_gravity=None
    ):
        log_luminosity_distance_gw_interp = (
            self.construct_luminosity_distance_GW_interpolation()
        )
        # Use linear interp: modified d_L_gw(z) table may be non-monotonic.
        return interp_z_in_log_distance_over_H0(
            luminosity_distance_gw,
            log_luminosity_distance_gw_interp,
            self.log_z_interp,
            method="linear",
            H0=1.0,  # H0 never appears in this model's dL_GW
            **self.interpolation_kwargs,
        )

    def get_dluminosity_distance_gw_over_dz_from_redshift(
        self, redshift, params_modified_gravity=None
    ):
        return self.distance_fun_jacobian(redshift, self.params_modified_gravity)

    def get_ratio_from_redshift(self, redshift):
        luminosity_distance_GW = self.distance_fun(
            redshift, self.params_modified_gravity
        )
        luminosity_distance = self.get_luminosity_distance_from_redshift(redshift)
        return luminosity_distance_GW / luminosity_distance


def get_luminosity_distance_GW_function_and_jacobian(name):
    if name == "FlatLambdaCDM_gp_integrated_dLGW":
        luminosity_distance_GW_function = luminosity_distance_GW_function_gp_nodes
        luminosity_distance_GW_function_jacobian = (
            luminosity_distance_GW_function_gp_nodes_jacobian
        )
    else:
        raise ValueError(f"Unknown luminosity distance GW model name: {name}")

    return luminosity_distance_GW_function, luminosity_distance_GW_function_jacobian


def luminosity_distance_GW_function_gp_nodes(z, params_modified_gravity):
    z_nodes = params_modified_gravity["z_nodes"]
    dLGW_nodes = params_modified_gravity["luminosity_distance_GW_nodes"]
    return jnp.interp(z, z_nodes, dLGW_nodes)


# TODO: better implementation?
def luminosity_distance_GW_function_gp_nodes_jacobian(z, params_modified_gravity):
    f_grad = jax.grad(luminosity_distance_GW_function_gp_nodes, argnums=0)

    def fz(zz):
        return f_grad(zz, params_modified_gravity)

    return jnp.vectorize(fz)(z)


# def power_spectrum_modified_gravity_gp(k, amplitude, correlation_scale):
#     return amplitude / (correlation_scale + (k ** 2) ** 1.3)


def power_spectrum_modified_gravity_gp(k, amplitude, correlation_scale):
    k_norm = safe_sqrt(k**2) * correlation_scale
    alpha_1 = 0.0
    alpha_2 = 4.0
    return (
        correlation_scale ** (-1)
        * amplitude
        * k_norm**alpha_1
        / (1 + (k_norm**alpha_2))
    )


def _create_field(num_bins):
    """Create a fresh RealField instance."""
    box_range = jnp.array([[0.0, 1.0]])
    box_shape_d = [num_bins]
    return field.RealField(
        box_range_d=box_range,
        box_shape_d=box_shape_d,
        power_spectrum_of_k=power_spectrum_modified_gravity_gp,
        replace_FT_with_packing=False,
    )


def softplus(x):
    return jnp.log(1 + jnp.exp(x))


def soft_low_clip(x, low_threshold, sigma=0.1):
    return low_threshold + softplus((x - low_threshold) / sigma) * sigma


def construct_ratio_gp_nodes(analysis, params, key_name="ratio", use_cumulative=True):

    cosmology_numerics = analysis.kwargs_analysis["cosmology_numerics"]

    params_ratio = params["modified_ratio"]
    gaussian_F_whitened_spatial = params_ratio[f"{key_name}_gaussian_whitened_field"]
    num_basis = gaussian_F_whitened_spatial.shape[-1]
    num_safe = num_basis - 10

    z = jnp.linspace(
        cosmology_numerics["z_min"], cosmology_numerics["z_max"], num_safe + 1
    )
    centers_z, delta_z = compute_centers_and_delta_from_array(z)

    if use_cumulative:
        cosmological_model_GR = FlatLambdaCDM(
            params=params["cosmology"], numerics=cosmology_numerics
        )
        luminosity_distance_GR = (
            cosmological_model_GR.get_luminosity_distance_from_redshift(z)
        )
        d_luminosity_distance_GR_over_dz = (
            luminosity_distance_GR[1:] - luminosity_distance_GR[:-1]
        ) / delta_z

    field_instance = _create_field(num_basis)
    field_instance.set_gaussian_F_whitened_from_gaussian_F_whitened_spatial(
        gaussian_F_whitened_spatial
    )

    params_power_spectrum = dict(
        amplitude=params_ratio[f"{key_name}_power_spectrum_amplitude"],
        correlation_scale=params_ratio.get(
            f"{key_name}_power_spectrum_correlation_scale"
        ),
    )
    field_instance.compute_gaussian_F_spatial_from_gaussian_F_whitened(
        power_spectrum_kwargs=params_power_spectrum, offset=None
    )

    # force integrated field to be 1 at low z
    supression_scale = params_ratio.get("supression_scale", None)
    supression_factor = 1 - jnp.exp(-((centers_z / supression_scale) ** 2))

    if use_cumulative:
        # take square to ensure positivity
        field_value = field_instance.gaussian_F_spatial[:num_safe]
        # smoothly keep above low_threshold
        field_value = soft_low_clip(field_value, low_threshold=-0.95)
        luminosity_distance_MG = jnp.cumsum(
            (1 + field_value * supression_factor)
            * d_luminosity_distance_GR_over_dz
            * delta_z
        )

        integrated_field = luminosity_distance_MG / luminosity_distance_GR[1:]
        integrated_field = jnp.concatenate([integrated_field[:1], integrated_field])

    else:
        field_value = (
            1
            + supression_factor
            * field_instance.gaussian_F_spatial[:num_safe]
            / (1 + z) ** 2
        )
        # smoothly bring field below 0.9 to 0.9 to prevent pathologies in HMC sampling from non-monotonic d_L_GW(z) at low z
        integrated_field = field_value

    return z, field_value, integrated_field


def complete_params_modified_gravity(analysis=None, params=None, kwargs_analysis=None):
    if kwargs_analysis is not None:
        analysis = SimpleNamespace(kwargs_analysis=dict(kwargs_analysis))
    if analysis is None:
        raise ValueError("Either analysis or kwargs_analysis must be provided.")
    if params is None:
        params = {}

    cosmology_numerics = dict(z_min=Z_MIN, z_max=Z_MAX, z_steps=Z_STEPS)
    cosmology_numerics.update(analysis.kwargs_analysis.get("cosmology_numerics", {}))
    analysis.kwargs_analysis["cosmology_numerics"] = cosmology_numerics

    if "cosmology" not in params:
        params["cosmology"] = dict(H0=70.0, Omega_m=0.3)

    name_modified_ratio = analysis.kwargs_analysis.get("cosmology_model_name")

    if name_modified_ratio in [
        "FlatLambdaCDM_GW_distance_cosine",
        "FlatLambdaCDM_GW_distance_gp_integrated",
        "FlatLambdaCDM_GW_distance_cM",
    ]:
        complete_ratio_params(analysis, params)
    elif name_modified_ratio in ["FlatLambdaCDM"]:
        pass
    else:
        raise NotImplementedError(
            f"Complete params not implemented for model {name_modified_ratio}"
        )

    return params


def complete_ratio_params(analysis, params):

    params_ratio = params["modified_ratio"]

    kwargs_analysis = analysis.kwargs_analysis
    name_modified_ratio = kwargs_analysis.get("cosmology_model_name")

    if name_modified_ratio == "FlatLambdaCDM_GW_distance_gp_integrated":
        params_ratio.setdefault("ratio_power_spectrum_correlation_scale", 1.0)
        params_ratio.setdefault("supression_scale", 0.1)
        params_ratio["z_nodes"], _, params_ratio["ratio_nodes"] = (
            construct_ratio_gp_nodes(
                analysis,
                params,
                key_name="ratio",
            )
        )
    elif name_modified_ratio in [
        "FlatLambdaCDM_GW_distance_cosine",
        "FlatLambdaCDM_GW_distance_cM",
    ]:
        pass
    else:
        raise NotImplementedError(
            f"Complete ratio params not implemented for model {name_modified_ratio}"
        )

    return params_ratio


def check_dGW_monotonicity_constraint(
    cosmological_model, params_modified_gravity, H0, z_check=None, penalty_scale=1e4
):
    """
    Smooth penalty for non-monotonic d_GW(z). Returns 0 when fully
    monotonic, smooth negative log-penalty otherwise.
    Unlike a hard -inf constraint, this gives HMC usable gradients
    near the monotonicity boundary, preventing pathological sampling.
    """
    z_check = jnp.linspace(
        0.0001, 2.0, 30
    )  # create a default redshift grid from Z_MIN to Z_MAX with 30 points
    #     else:
    #         nbins = int(float(cosmology_numerics['z_max']) * 5)
    # z_check = jnp.linspace(cosmology_numerics['z_min'], cosmology_numerics['z_max'], cosmology_numerics['bins_for_monotonicity'])
    dGW_vals = cosmological_model.get_luminosity_distance_gw_from_redshift(
        z_check, H0, params_modified_gravity
    )  # evaluate the GW luminosity distance d_GW(z) on the grid

    dz = (
        z_check[1:] - z_check[:-1]
    )  # compute interval widths between consecutive redshift points
    ddGW_dz_approx = (
        dGW_vals[1:] - dGW_vals[:-1]
    ) / dz  # finite-difference approximation to d(d_GW)/dz on each interval

    # Smooth quadratic penalty for negative slopes.
    # Normalize by the typical derivative magnitude for numerical stability.
    scale = (
        jnp.max(jnp.abs(ddGW_dz_approx)) + 1e-10
    )  # max absolute derivative used to scale/normalize; add tiny eps to avoid division by zero
    normalized_deriv = (
        ddGW_dz_approx / scale
    )  # normalized derivative values (order-unity)
    violations = jnp.minimum(
        normalized_deriv, 0.0
    )  # keep only negative (non-monotonic) parts; positive slopes become 0

    return (
        -penalty_scale * jnp.sum(violations**2)
    )  # return negative quadratic penalty (0 if monotonic, negative if violations exist)
