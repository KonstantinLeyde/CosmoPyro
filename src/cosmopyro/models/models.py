from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpyro

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

from ..cosmology.modified_gw_distance_ratio import (
    complete_params_modified_gravity,
)
from ..distributions.grid_distributions import (
    InterpolatedConditional1D,
    SeparableConditional1D,
    normalize_cond_interpolated_1d,
)
from ..numpyro_utils.batching import (
    apply_batched_operation_1d,
)
from ..numpyro_utils.sampling_utils import (
    get_prior_draw,
)

# this also registers the pytree for SimpleNamespace
from ..utils.jax_utils import (
    get_penalty_factor_relative_variance,
    get_relative_variance,
    handle_nonfinite_log_probs,
    log_estimator_and_variance,
    log_estimator_and_variance_stacked,
)

__all__ = [
    "calculate_safe_log_prob_batched",
    "get_log_prob",
    "log_jacobian_md1md2dL_to_msmass_ratioz",
    "model_evaluate_p_theta",
]


def model_evaluate_p_theta(
    analysis,
    data,
    use_value_for_Delta=True,
    grid_ref=None,
    skymap=None,
):

    n = data.samples.num_posterior_samples_per_event
    num_events = n.shape[0]

    params = get_prior_draw(analysis.prior, use_value_for_Delta=use_value_for_Delta)
    params["cosmology"]["H0"] = params["cosmology"]["h"] * 100

    params = complete_params_modified_gravity(analysis, params)

    cosmological_model = analysis.get_cosmological_model(
        parameters=params,
    )

    model_mass_dim1, model_mass_dim2 = analysis.get_source_frame_mass_model(params)

    marginal_redshift_model = analysis.get_redshift_model(
        analysis.get_distribution_name("redshift"),
        cosmological_model=cosmological_model,
        parameters=params,
    )

    # distribution redshift-skyposition
    x_bins = {p: analysis.binning["boundaries"][p] for p in ["redshift"]}
    y_bins = {p: analysis.binning["boundaries"][p] for p in ["healpix_idx"]}

    centers_redshift = analysis.binning["centers"]["redshift"]

    log_prob_redshift_nn = marginal_redshift_model.log_prob(
        dict(redshift=centers_redshift)
    )
    prob_redshift = normalize_cond_interpolated_1d(
        x_edges=analysis.binning["boundaries"]["redshift"],
        cond=jnp.exp(log_prob_redshift_nn),
    )

    # P(z | sky position) ∝ p(z) * skymap(z, pix); the per-pixel normalization
    # is a single matvec inside SeparableConditional1D instead of building and
    # normalizing the full (nz, npix) product grid every likelihood call.
    prob_skyposition = skymap.prob_skyposition_zhp if skymap is not None else None
    model_redshift_G_skyposition = SeparableConditional1D(
        x_bins=x_bins,
        y_bins=y_bins,
        prob_x=prob_redshift,
        grid=prob_skyposition,
    )

    # define redshift model only marginally (no dependence on skyposition)
    model_redshift = InterpolatedConditional1D(
        x_bins=x_bins, y_bins=None, cond=prob_redshift
    )

    distributions = {
        0: model_mass_dim1,
        1: model_mass_dim2,
        2: model_redshift_G_skyposition,
    }

    if (
        analysis.kwargs_analysis.get("injection_evaluation", "exclude_skyposition")
        == "exclude_skyposition"
    ):
        # define same distribution for injections
        distributions_inj = {
            0: model_mass_dim1,
            1: model_mass_dim2,
            2: model_redshift,
        }
    else:
        distributions_inj = distributions

    # evaluate batched, to avoid OOM GPU issues
    log_prob_flat = calculate_safe_log_prob_batched(
        data.samples,
        cosmological_model,
        distributions,
        params=params,
        batch_size=analysis.get_batch_size("posterior_samples"),
    )
    log_prob_flat = handle_nonfinite_log_probs(
        log_prob_flat, analysis, "posterior_samples"
    )
    # No default: if this were silently assumed True for variable group sizes, the
    # uniform path would reshape the stacked weights into equal blocks and return a
    # wrong likelihood with a perfectly finite gradient. The data loaders always set it.
    posterior_sample_groups_are_uniform = (
        data.samples.posterior_sample_groups_are_uniform
    )
    posterior_sample_segment_ids = (
        None
        if posterior_sample_groups_are_uniform
        else getattr(data.samples, "posterior_sample_segment_ids", None)
    )
    log_prob_E, log_variances_E = log_estimator_and_variance_stacked(
        log_prob_flat,
        n,
        uniform_groups=posterior_sample_groups_are_uniform,
        segment_ids=posterior_sample_segment_ids,
    )

    log_prob_inj = calculate_safe_log_prob_batched(
        data.injections,
        cosmological_model,
        distributions_inj,
        params=params,
        batch_size=analysis.get_batch_size("injections"),
    )
    log_prob_inj = handle_nonfinite_log_probs(
        log_prob_inj, analysis, "injections"
    ) - jnp.log(data.injections.num_events)

    log_prob_selection, log_variances_selection = log_estimator_and_variance(
        log_prob_inj
    )

    numpyro.factor("log_prob_E", log_prob_E)
    numpyro.factor("log_prob_selection", -log_prob_selection * num_events)

    log_relative_variance_E = get_relative_variance(log_prob_E, log_variances_E)
    log_relative_variance_selection = get_relative_variance(
        log_prob_selection, log_variances_selection
    )

    likelihood_eval_kwargs = analysis.kwargs_analysis.get("likelihood_evaluation", {})
    if likelihood_eval_kwargs.get("save_effective_sample_sizes", False):
        numpyro.deterministic("log_variances_E", log_variances_E)

    if likelihood_eval_kwargs.get("penalty_factor_relative_variance", False):
        # This is a hack to punish high relative variance in the likelihood estimator, which can cause issues for HMC sampling.
        penalty_strength = likelihood_eval_kwargs.get(
            "penalty_factor_relative_variance", 1.0
        )
        penalty_factor_relative_variance = get_penalty_factor_relative_variance(
            log_relative_variance_E, log_relative_variance_selection, penalty_strength
        )
        numpyro.factor(
            "penalty_factor_relative_variance", penalty_factor_relative_variance
        )

    # for post-processing
    if grid_ref is not None:
        dim1_name = model_mass_dim1.x_var

        if dim1_name == "mass_1_s":
            # old (mass_1_s, mass_ratio) parameterization
            v1 = grid_ref["mass_1_s"]
            v2 = grid_ref["mass_ratio"]
            shape = (v1.shape[0], v2.shape[0])
            x1 = {"mass_1_s": jnp.broadcast_to(v1[:, None], shape)}
            x2 = {"mass_ratio": jnp.broadcast_to(v2[None, :], shape)}
            y2 = {"mass_1_s": jnp.broadcast_to(v1[:, None], shape)}
        else:
            # new (log_mass_total_s, minus_log_mass_ratio) parameterization
            m1s = grid_ref["mass_1_s"]
            q = grid_ref["mass_ratio"]
            shape = (m1s.shape[0], q.shape[0])
            s_grid = jnp.log(m1s[:, None] * (1 + q[None, :]))
            d_grid = -jnp.log(q[None, :]) * jnp.ones_like(m1s[:, None])
            x1 = {"log_mass_total_s": s_grid}
            x2 = {"minus_log_mass_ratio": jnp.broadcast_to(d_grid, shape)}
            y2 = {"log_mass_total_s": s_grid}

        log_prob_dim1 = model_mass_dim1.log_prob(x1)
        log_prob_dim2 = model_mass_dim2.log_prob(x2, y2)
        log_prob_mass_1_s_mass_ratio = log_prob_dim1 + log_prob_dim2

        if dim1_name != "mass_1_s":
            # Convert density from (s, delta) to (m1s, q) space for visualization:
            # p(m1s, q) = p(s, delta) * |d(s,delta)/d(m1s,q)| = p(s, delta) / (m1s * q)
            m1s = grid_ref["mass_1_s"]
            q = grid_ref["mass_ratio"]
            log_prob_mass_1_s_mass_ratio = (
                log_prob_mass_1_s_mass_ratio
                - jnp.log(m1s[:, None])
                - jnp.log(q[None, :])
            )

        numpyro.deterministic(
            "log_prob_mass_1_s_mass_ratio", log_prob_mass_1_s_mass_ratio
        )

        # redshift and sky position p(z, sky position) or p(z)

        compute_full_redshift_skyposition = False

        if compute_full_redshift_skyposition:
            redshift = grid_ref["redshift"]
            healpix_idx = grid_ref["healpix_idx"]

            redshift_m, healpix_idx_m = jnp.meshgrid(
                redshift, healpix_idx, indexing="ij"
            )

            log_prob_redshift_skyposition = model_redshift_G_skyposition.log_prob(
                {
                    "redshift": redshift_m,
                },
                {
                    "healpix_idx": healpix_idx_m,
                },
            )
            # can use this because sky position is uniform
            numpyro.deterministic(
                "log_prob_redshift_skyposition", log_prob_redshift_skyposition
            )

        else:
            redshift = grid_ref["redshift"]
            log_prob = model_redshift.log_prob({"redshift": redshift})
            numpyro.deterministic("log_prob_redshift", log_prob)

        if analysis.kwargs_analysis["cosmology_model_name"] not in ["FlatLambdaCDM"]:
            ratio = cosmological_model.get_ratio_from_redshift(redshift)
            numpyro.deterministic("ratio_luminosity_distance_gw_em", ratio)


def get_log_prob(samples, cosmological_model, distributions, params):

    redshift = cosmological_model.get_redshift_from_luminosity_distance_gw(
        luminosity_distance_gw=samples.luminosity_distance,
    )

    mass_1_s = samples.mass_1_d / (1 + redshift)
    mass_ratio = samples.mass_ratio
    mass_2_s = mass_1_s * mass_ratio

    data = {
        "mass_1_s": mass_1_s,
        "mass_ratio": mass_ratio,
        "log_mass_total_s": jnp.log(mass_1_s + mass_2_s),
        "minus_log_mass_ratio": -jnp.log(mass_ratio),
        "redshift": redshift,
        "healpix_idx": samples.healpix_idx,
    }

    log_prob_parts = []
    uses_s_delta = False
    for k in distributions:
        dist = distributions[k]
        x_vals = {p: data[p] for p in dist.x_names}
        y_vals = {p: data[p] for p in dist.y_names}
        log_prob_part = dist.log_prob(x_vals=x_vals, y_vals=y_vals)
        if "log_mass_total_s" in dist.x_names or "minus_log_mass_ratio" in dist.x_names:
            uses_s_delta = True

        log_prob_parts.append(log_prob_part)

    dim_log = log_prob_parts[0].ndim
    log_prob_parts = jnp.stack(log_prob_parts, axis=-dim_log - 1)
    log_prob = jnp.sum(log_prob_parts, axis=-dim_log - 1)

    log_jacobian = log_jacobian_md1md2dL_to_msmass_ratioz(
        cosmological_model, redshift, samples.mass_1_d
    )
    if uses_s_delta:
        # Additional Jacobian for (m1_s, q) -> (log_mass_total_s, minus_log_mass_ratio)
        # |d(s, delta)/d(m1_s, q)| = 1 / (m1_s * q)
        log_jacobian = log_jacobian + jnp.log(mass_1_s) + jnp.log(mass_ratio)

    return log_prob - log_jacobian - jnp.log(samples.prior_masses_d_dL)


_SAMPLE_FIELDS = (
    "luminosity_distance",
    "mass_1_d",
    "mass_ratio",
    "healpix_idx",
    "prior_masses_d_dL",
)


def calculate_safe_log_prob_batched(
    all_samples, cosmological_model, distributions, params, batch_size=1000000
):
    """
    Flattens (events, samples) -> 1D, computes log_prob in batches,
    and reshapes back to (events, samples).
    """

    # A. Determine Shapes & Flatten Inputs
    ref_array = all_samples.luminosity_distance
    original_shape = ref_array.shape
    total_elements = ref_array.size

    def flatten_to_1d(x):
        return jnp.reshape(x, (total_elements,))

    # Strip metadata fields (e.g. num_posterior_samples_per_event) that have
    # a different shape and are not used by get_log_prob.
    sample_data = SimpleNamespace(
        **{
            k: getattr(all_samples, k)
            for k in _SAMPLE_FIELDS
            if hasattr(all_samples, k)
        }
    )

    if len(original_shape) == 1:
        # Already 1D, no need to flatten
        flat_samples = sample_data
    else:
        flat_samples = jax.tree_util.tree_map(flatten_to_1d, sample_data)

    # B. Define the Operation for a Single Batch
    # This wrapper binds the static arguments so the batcher only sees 'samples'
    def batch_op(samples_batch):
        return get_log_prob(samples_batch, cosmological_model, distributions, params)

    # C. Execute Batching
    flat_result = apply_batched_operation_1d(flat_samples, batch_op, batch_size)

    # D. Reshape to Original Dimensions
    return jnp.reshape(flat_result, original_shape)


def log_jacobian_md1md2dL_to_msmass_ratioz(cosmological_model, redshift, mass_1_d):
    r"""
    Log Jacobian for transformation from (mass_1_d, mass_2_d, luminosity_distance) to (mass_1_s, mass_ratio, redshift):

    \log |d(mass_1_d, mass_2_d, dL) / d(mass_1_s, mass_ratio, z)| = \log(dL/dz) + \log(1+z) + \log(mass_1_s)

    Arguments:
        cosmological_model: Cosmological model to compute dL/dz
        redshift: Redshift array
        mass_1_d: Detector-frame mass 1 array (used to compute mass_1_s = mass_1_d / (1+z))

    Returns:
        log_jacobian: Logarithm of the Jacobian determinant for the transformation

    """

    term_1 = 1 + redshift
    term_2 = mass_1_d
    term_3 = cosmological_model.get_dluminosity_distance_gw_over_dz_from_redshift(
        redshift
    )

    return jnp.log(term_1) + jnp.log(term_2) + jnp.log(term_3)
