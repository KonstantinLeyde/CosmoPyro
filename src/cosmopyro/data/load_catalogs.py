from types import SimpleNamespace

import jax
import jax.numpy as jnp

from ..utils.jax_utils import (
    group_sizes_are_uniform,
    make_segment_ids_from_group_sizes,
)
from ..utils.spherical_coordinates import HealPixDiscretization
from .data_utils import (
    load_hdf5_to_namespace,
)

__all__ = [
    "DEFAULT_INJECTION_KEYS",
    "add_healpix_idx_to_simple_namespace_or_randomly_draw",
    "add_in_fake_posterior_samples_dimension",
    "add_missing_variables_to_samples",
    "cut_injections",
    "flatten_samples_to_stacked",
    "get_healpix_idx_from_simple_namespace",
    "get_posterior_samples_down_selected",
    "load_posterior_samples_and_injections_from_file",
    "prepare_injections",
]


def load_posterior_samples_and_injections_from_file(
    filepath,
    filepath_injections,
    num_events,
    num_posterior_samples,
    num_injections=None,
    nside=None,
    seed=0,
):

    data = SimpleNamespace()

    samples = load_hdf5_to_namespace(filepath)
    data.injections = prepare_injections(
        filepath_injections, keys=None, num_injections=num_injections
    )

    if nside is not None:
        add_healpix_idx_to_simple_namespace_or_randomly_draw(
            samples, nside=nside, seed=seed
        )
        add_healpix_idx_to_simple_namespace_or_randomly_draw(
            data.injections, nside=nside, seed=seed + 1
        )

    if (
        samples.mass_1_d.ndim == 1
        and getattr(samples, "num_posterior_samples_per_event", None) is not None
    ):
        print("Samples already in flattened format. ")
        data.samples = samples
        data.samples.posterior_sample_groups_are_uniform = group_sizes_are_uniform(
            data.samples.num_posterior_samples_per_event
        )

        if num_posterior_samples is not None or num_events is not None:
            raise ValueError(
                "Cannot specify num_posterior_samples or num_events when samples are already in flattened format with num_posterior_samples_per_event. "
            )

        if (
            not data.samples.posterior_sample_groups_are_uniform
            and getattr(data.samples, "posterior_sample_segment_ids", None) is None
        ):
            data.samples.posterior_sample_segment_ids = (
                make_segment_ids_from_group_sizes(
                    data.samples.num_posterior_samples_per_event
                )
            )
    else:
        data.samples = get_posterior_samples_down_selected(
            samples, num_events, num_posterior_samples
        )
        data.samples = flatten_samples_to_stacked(data.samples)

    return data


def add_healpix_idx_to_simple_namespace_or_randomly_draw(
    data_ns, nside, scheme="ring", seed=0
):

    if "healpix_idx" in vars(data_ns):
        print("healpix_idx already exists in data_ns, skipping adding healpix_idx.")
        return

    if "ra" in vars(data_ns) and "dec" in vars(data_ns):
        print(f"Adding healpix_idx from ra, dec, nside={nside}, scheme={scheme}.")

        data_ns.healpix_idx = get_healpix_idx_from_simple_namespace(
            data_ns, nside=nside, scheme=scheme
        )
    else:
        print(f"Adding random healpix_idx to injections, nside={nside}.")

        npix = 12 * nside**2
        data_ns.healpix_idx = jax.random.randint(
            jax.random.PRNGKey(seed + 1), data_ns.mass_1_d.shape, 0, npix
        )


def get_healpix_idx_from_simple_namespace(data_ns, nside, scheme="ring"):

    healpix_discretization = HealPixDiscretization(nside=nside, scheme=scheme)
    theta = jnp.pi / 2 - data_ns.dec
    phi = data_ns.ra

    if theta.ndim == 2:
        return healpix_discretization.ang2pixvv(theta, phi)
    elif theta.ndim == 1:
        return healpix_discretization.ang2pixv(theta, phi)
    else:
        raise ValueError(f"Unsupported ndim for theta: {theta.ndim}")


def get_posterior_samples_down_selected(
    samples, num_events=None, num_posterior_samples=None
):

    s = samples.mass_1_d.shape

    if len(s) == 1:
        print("Warning, passing 1D samples, assuming this is a test for true values. ")
        if num_posterior_samples is not None and num_posterior_samples != 1:
            raise ValueError(
                f"For 1D samples, num_posterior_samples must be 1, but got {num_posterior_samples}"
            )
        samples = add_in_fake_posterior_samples_dimension(samples)
        s = samples.mass_1_d.shape
    elif len(s) != 2:
        raise ValueError(f"Unsupported shape for samples: {s}")

    if num_events is None:
        num_events = s[0]
    if num_posterior_samples is None:
        num_posterior_samples = s[1]

    if s[0] < num_events:
        raise ValueError(
            f"Requested num_events {num_events} exceeds available samples {s[0]}"
        )
    if s[1] < num_posterior_samples:
        raise ValueError(
            f"Requested num_posterior_samples {num_posterior_samples} exceeds available samples {s[1]}"
        )

    samples_cut = SimpleNamespace()

    for attr in [
        "mass_1_d",
        "chirp_mass_d",
        "mass_ratio",
        "luminosity_distance",
        "healpix_idx",
        "prior_masses_d_dL",
    ]:
        if hasattr(samples, attr):
            v = getattr(samples, attr)[:num_events, :num_posterior_samples]
            setattr(samples_cut, attr, v)
        else:
            print(f"Warning: attribute {attr} not found in samples, skipping.")

    return samples_cut


DEFAULT_INJECTION_KEYS = [
    "mass_1_d",
    "mass_ratio",
    "luminosity_distance",
    "healpix_idx",
    "prior_masses_d_dL",
    "ra",
    "dec",
]


def prepare_injections(filepath, keys=None, num_injections=None):
    """Load injection set from HDF5, compute derived fields, and optionally subsample."""
    if keys is None:
        keys = DEFAULT_INJECTION_KEYS

    injections_raw = load_hdf5_to_namespace(filepath)

    # Derive prior_masses_d_dL from log_prob if not stored directly
    if not hasattr(injections_raw, "prior_masses_d_dL"):
        if not hasattr(
            injections_raw, "log_prob_mass_1_d_mass_2_d_luminosity_distance"
        ):
            raise ValueError(
                "Injection file must contain either 'prior_masses_d_dL' or "
                "'log_prob_mass_1_d_mass_2_d_luminosity_distance'."
            )
        log_prob_draw = injections_raw.log_prob_mass_1_d_mass_2_d_luminosity_distance
        injections_raw.prior_masses_d_dL = jnp.exp(
            log_prob_draw - jnp.max(log_prob_draw)
        )

    # num_events = total number of drawn signals (before SNR cut)
    if hasattr(injections_raw, "num_total_injections"):
        injections_raw.num_events = int(jnp.sum(injections_raw.num_total_injections))
    elif not hasattr(injections_raw, "num_events"):
        raise ValueError(
            "Injection file must contain 'num_total_injections' or 'num_events'. "
            "This is the total number of simulated signals (before selection), "
            "not the number of detected injections."
        )

    # Select only the required keys (avoids issues with batching scalar attributes)
    injections = SimpleNamespace()
    for attr in keys:
        if hasattr(injections_raw, attr):
            setattr(injections, attr, getattr(injections_raw, attr))
        else:
            print(f"Warning: attribute {attr} not found in injections_raw, skipping.")

    injections.num_events = injections_raw.num_events

    if num_injections is not None:
        injections = cut_injections(
            injections, keys=keys, num_injections=num_injections
        )

    return injections


def cut_injections(injections, keys, num_injections):

    if num_injections is None:
        return injections

    num_injections = int(num_injections)

    if injections.mass_1_d.shape[0] < num_injections:
        raise ValueError(
            f"Requested num_injections {num_injections} exceeds available injections {injections.mass_1_d.shape[0]}"
        )

    injections_cut = SimpleNamespace()

    for attr in keys:
        if hasattr(injections, attr):
            v = getattr(injections, attr)[:num_injections]
            setattr(injections_cut, attr, v)
        else:
            print(f"Warning: attribute {attr} not found in injections, skipping.")

    # take the relative portion of the injections
    ratio = num_injections / injections.mass_1_d.shape[0]
    injections_cut.num_events = int(injections.num_events * ratio)

    if injections_cut.num_events == 0:
        raise ValueError(
            f"After cutting injections, num_events is 0. Requested num_injections {num_injections} is too small relative to total injections {injections.mass_1_d.shape[0]} and total events {injections.num_events}. Consider increasing num_injections."
        )

    print("Ratio of injections kept: ", ratio)

    return injections_cut


def flatten_samples_to_stacked(samples):
    """Flatten 2D (num_events, num_posterior_samples) samples to 1D stacked format.

    Adds a ``num_posterior_samples_per_event`` attribute (1D array of group sizes).
    """
    ref_shape = samples.mass_1_d.shape
    if len(ref_shape) != 2:
        raise ValueError(f"Expected 2D samples, got shape {ref_shape}")

    num_events, num_posterior_samples = ref_shape
    # creates an array of shape (num_events,) where each entry is num_posterior_samples
    n = jnp.full(num_events, num_posterior_samples, dtype=jnp.int64)

    flat_samples = SimpleNamespace()
    for attr in vars(samples):
        v = getattr(samples, attr)
        if hasattr(v, "shape") and v.shape == ref_shape:
            setattr(flat_samples, attr, v.reshape(-1))

    flat_samples.num_posterior_samples_per_event = n
    flat_samples.posterior_sample_groups_are_uniform = True
    return flat_samples


def add_in_fake_posterior_samples_dimension(samples):

    samples_new = SimpleNamespace()

    for attr in [
        "mass_1_d",
        "chirp_mass_d",
        "mass_ratio",
        "luminosity_distance",
        "healpix_idx",
    ]:
        if hasattr(samples, attr):
            v = getattr(samples, attr)
            setattr(samples_new, attr, v[:, None])
        else:
            print(f"Warning: attribute {attr} not found in samples, skipping.")

    samples_new.prior_masses_d_dL = jnp.ones_like(samples_new.mass_1_d)

    return samples_new


def add_missing_variables_to_samples(samples, cosmological_model=None):

    if cosmological_model is None:
        print(
            "No cosmological model provided, using default FlatLambdaCDM with h=0.7, Omega_m=0.3."
        )

        from ..cosmology.flat_lambdacdm import FlatLambdaCDM

        cosmological_model = FlatLambdaCDM(params=dict(h=0.7, Omega_m=0.3))

    if not hasattr(samples, "mass_2_d"):
        samples.mass_2_d = samples.mass_1_d * samples.mass_ratio

    if not hasattr(samples, "log_mass_total_s"):
        samples.log_mass_total_s = jnp.log(samples.mass_1_d + samples.mass_2_d)

    if not hasattr(samples, "minus_log_mass_ratio"):
        samples.minus_log_mass_ratio = -jnp.log(samples.mass_2_d / samples.mass_1_d)

    if not hasattr(samples, "redshift"):
        samples.redshift = cosmological_model.get_redshift_from_luminosity_distance(
            samples.luminosity_distance
        )

    if not hasattr(samples, "mass_1_s"):
        samples.mass_1_s = samples.mass_1_d / (1 + samples.redshift)

    if not hasattr(samples, "mass_2_s"):
        samples.mass_2_s = samples.mass_2_d / (1 + samples.redshift)

    if not hasattr(samples, "chirp_mass_s"):
        samples.chirp_mass_s = (samples.mass_1_s * samples.mass_2_s) ** (3 / 5) / (
            samples.mass_1_s + samples.mass_2_s
        ) ** (1 / 5)

    if not hasattr(samples, "chirp_mass_d"):
        samples.chirp_mass_d = (samples.mass_1_d * samples.mass_2_d) ** (3 / 5) / (
            samples.mass_1_d + samples.mass_2_d
        ) ** (1 / 5)
