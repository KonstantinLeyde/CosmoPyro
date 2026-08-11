from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from ..utils.transformations import (
    get_jacobian_chirp_mass_mass_ratio_from_component_masses,
    get_mass_1_from_chirp_mass_and_mass_ratio,
)
from .generate_catalogs import optimal_snr_approximated

__all__ = [
    "apply_delta_uncertainties",
    "apply_gamma_uncertainties",
    "apply_gamma_uncertainties_batched",
    "compute_posterior_samples",
    "draw_gamma_observation",
    "get_iota_prior_samples",
    "get_samples_from_gamma_likelihood",
    "get_unit_interval_samples_from_gamma_likelihood",
    "resample_idx_from_p",
    "resample_on_finite_domain",
    "resolve_concentrations",
    "reweigh_samples_with_snr_contribution",
    "snr_coefficient",
    "warn_on_low_reweighting_ess",
]


def _as_key(seed_or_key):
    """Accept either an integer seed or an already-split PRNG key.

    Every random stream below is derived by splitting.
    """
    if isinstance(seed_or_key, (int, np.integer)):
        return jax.random.PRNGKey(int(seed_or_key))
    return seed_or_key


def compute_posterior_samples(
    data_all, num_events, num_posterior_samples, uncertainties, seed, concentration=None
):

    samples_true = SimpleNamespace()
    # select first num_events events
    for attr in [
        "mass_1_d",
        "chirp_mass_d",
        "mass_ratio",
        "luminosity_distance",
        "scatter_snr",
    ]:
        v = getattr(data_all, attr)[:num_events]
        setattr(samples_true, attr, v)

    # propagate also config
    samples_true.config = data_all.config

    if uncertainties == "delta":
        samples = apply_delta_uncertainties(samples_true)
    elif uncertainties == "gamma":
        samples = apply_gamma_uncertainties_batched(
            samples_true,
            num_posterior_samples=num_posterior_samples,
            concentration=concentration,
            seed=seed**3 + 41,
            batch_size=10,
        )
    else:
        raise ValueError(f"Unrecognized uncertainties type: {uncertainties}")

        # if the prior is not defined, assume flat
    if not hasattr(samples, "prior_masses_d_dL"):
        samples.prior_masses_d_dL = jnp.ones(samples.mass_1_d.shape)
        print("Warning: prior_masses_d_dL not found in samples, assuming flat prior.")

    return samples


def apply_delta_uncertainties(samples_true, num_posterior_samples=1):
    samples = SimpleNamespace()
    for attr in vars(samples_true):
        if attr == "config":
            continue

        v = getattr(samples_true, attr)[..., None]
        v = jnp.repeat(v, num_posterior_samples, axis=-1)
        setattr(samples, attr, v)
    return samples


def get_iota_prior_samples(num_events, num_posterior_samples, key):
    key = _as_key(key)
    u = jax.random.uniform(
        key, (num_events, num_posterior_samples), minval=-1.0, maxval=1.0
    )
    iota_samples = jnp.arccos(u)
    return iota_samples


def apply_gamma_uncertainties_batched(
    samples_true, num_posterior_samples, concentration=None, seed=34, batch_size=50
):
    """
    Apply apply_gamma_uncertainties in batches.
    Needed, because otherwise memory usage is too high.

    """

    sns = []
    num_events = samples_true.chirp_mass_d.shape[0]
    num_batch = (num_events + batch_size - 1) // batch_size
    base_key = _as_key(seed)
    for i in range(num_batch):
        print(f"Applying gamma uncertainties, batch {i + 1} / {num_batch}...")

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_events)

        samples_batch = SimpleNamespace()
        samples_batch.config = samples_true.config
        for attr in vars(samples_true):
            if attr == "config":
                continue
            v = getattr(samples_true, attr)[start_idx:end_idx]
            setattr(samples_batch, attr, v)
        sns.append(
            apply_gamma_uncertainties(
                samples_batch,
                num_posterior_samples,
                concentration,
                # fold_in gives each batch an independent stream; the old
                # `seed + i * 10` collided across batches (batch j+10's iota
                # key equalled batch j's resampling key).
                jax.random.fold_in(base_key, i),
            )
        )
    # concatenate batches
    samples_f = SimpleNamespace()
    for attr in vars(sns[0]):
        if attr == "config":
            continue
        v = jnp.concatenate([getattr(sn, attr) for sn in sns], axis=0)
        setattr(samples_f, attr, v)
    samples_f.config = samples_true.config
    return samples_f


def resolve_concentrations(concentration, num_events, key):
    """Expand any ``(low, high)`` entry into one Uniform draw per event.

    A scalar entry keeps its current meaning: every event measured equally well.
    A two-element tuple/list means the concentration itself varies event to event,
    drawn uniformly on [low, high] -- a cheap way to mimic a catalog with a mix of
    well- and poorly-measured events.

    Note the measurement precision is ``1 / sqrt(k + 1)``, so uniform in k is not
    uniform in precision; it weights the well-measured end.

    Anything that is neither a positive float nor a positive ``(low, high)`` pair
    raises, rather than being coerced into an array whose shape then broadcasts
    somewhere unhelpful.
    """
    resolved = {}
    # sorted() so the key assignment does not depend on dict insertion order
    names = sorted(concentration)
    keys = jax.random.split(_as_key(key), len(names))

    for key_name, name in zip(keys, names):
        value = concentration[name]

        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(
                    f"concentration['{name}'] = {value!r}: a sequence must be a "
                    f"(low, high) pair, got length {len(value)}."
                )
            low, high = (float(v) for v in value)
            if not high > low:
                raise ValueError(
                    f"concentration['{name}'] = {value!r} needs high > low."
                )
            if low <= 0.0:
                raise ValueError(
                    f"concentration['{name}'] = {value!r} must be positive."
                )
            resolved[name] = jax.random.uniform(
                key_name, (num_events,), minval=low, maxval=high
            )

        elif isinstance(
            value, (float, int, np.floating, np.integer)
        ) and not isinstance(value, bool):
            if float(value) <= 0.0:
                raise ValueError(
                    f"concentration['{name}'] = {value!r} must be positive."
                )
            resolved[name] = jnp.asarray(float(value))

        else:
            raise TypeError(
                f"concentration['{name}'] = {value!r} has type "
                f"{type(value).__name__}; expected a float (the same for every "
                "event) or a (low, high) tuple (drawn uniformly per event)."
            )

    return resolved


def apply_gamma_uncertainties(
    samples_true, num_posterior_samples, concentration=None, seed=34
):
    """
    Apply gamma uncertainties to the samples (around a measured quantity).
    Relative uncertainty is roughly 1/sqrt(concentration + 1).

    Each ``concentration`` entry is either a scalar (all events measured equally
    well) or a ``(low, high)`` tuple, in which case the concentration is drawn
    uniformly per event. The drawn values are returned as
    ``concentration_<parameter>`` so you can see which events were measured well.

    Each parameter is proposed from a distribution matched to its own target and
    the mismatch is carried by an importance weight, all of which is folded into
    the single resampling step at the end:

    * ``chirp_mass_d``: straight from its Gamma posterior, weight 1.
    * ``mass_ratio``: Gamma posterior truncated to (0, 1). The proposal rate is
      tilted so its mean matches the truncated target mean; the weight enforces
      the truncation. Rejecting instead would discard ~56% of proposals on
      average and >99% for events near equal mass.
    * ``luminosity_distance``: not proposed from its posterior. It is
      solved for from the SNR constraint, which makes the very sharp chi2 SNR
      factor cancel analytically. See ``reweigh_samples_with_snr_contribution``.

    The prior stays flat in (chirp_mass_d, mass_ratio, luminosity_distance), so
    ``prior_masses_d_dL`` is the usual ``chirp_mass / m1**2``.
    """

    if num_posterior_samples <= 1:
        raise ValueError("num_posterior_samples must be > 1 for gamma uncertainties")

    if concentration is None:
        print("Warning: concentration not provided, using default values.")
        concentration = {
            "chirp_mass_d": 1000.0,
            "mass_ratio": 50.0,
            "luminosity_distance": 25.0,
        }

    samples_sn = SimpleNamespace()

    buffer_factor = 200
    num_posterior_samples_prop = buffer_factor * num_posterior_samples

    # one stream each for: chirp mass, mass ratio, the dL observation, the SNR
    # step, and the per-event concentration draws
    keys = jax.random.split(_as_key(seed), 5)

    concentration = resolve_concentrations(
        concentration, jnp.shape(samples_true.chirp_mass_d)[0], keys[4]
    )

    samples_sn.chirp_mass_d = get_samples_from_gamma_likelihood(
        samples_true.chirp_mass_d,
        SimpleNamespace(
            concentration=concentration["chirp_mass_d"],
            prior_low=None,
            prior_high=None,
            num_posterior_samples=num_posterior_samples_prop,
        ),
        keys[0],
    )

    samples_sn.mass_ratio, log_weight_mass_ratio = (
        get_unit_interval_samples_from_gamma_likelihood(
            samples_true.mass_ratio,
            concentration["mass_ratio"],
            num_posterior_samples_prop,
            keys[1],
        )
    )

    luminosity_distance_likelihood = SimpleNamespace(
        concentration=concentration["luminosity_distance"],
        data=draw_gamma_observation(
            samples_true.luminosity_distance,
            concentration["luminosity_distance"],
            keys[2],
        ),
    )

    # complete mass_1_d and prior_masses_d_dL, the (chirp_mass, mass_ratio)
    # Jacobian matching the flat-in-(chirp_mass, mass_ratio) prior
    samples_sn.mass_1_d = get_mass_1_from_chirp_mass_and_mass_ratio(
        samples_sn.chirp_mass_d, samples_sn.mass_ratio
    )
    samples_sn.prior_masses_d_dL = (
        get_jacobian_chirp_mass_mass_ratio_from_component_masses(
            samples_sn.mass_1_d, samples_sn.mass_1_d * samples_sn.mass_ratio
        )
    )

    samples_f = reweigh_samples_with_snr_contribution(
        samples_sn,
        samples_true,
        num_posterior_samples,
        keys[3],
        luminosity_distance_likelihood,
        log_weight_extra=log_weight_mass_ratio,
    )

    # record what each event was actually measured with, broadcast to one value
    # per event so the batches concatenate along axis 0 like everything else
    num_events = jnp.shape(samples_true.chirp_mass_d)[0]
    for name, value in concentration.items():
        setattr(
            samples_f, f"concentration_{name}", jnp.broadcast_to(value, (num_events,))
        )

    return samples_f


def warn_on_low_reweighting_ess(p, num_posterior_samples, event_offset=0):
    """Report events where the SNR reweighting has too few effective samples.

    ``resample_idx_from_p`` draws with replacement, so a low effective sample
    size silently yields an event whose posterior is a handful of proposals
    repeated many times. Nothing downstream can tell that apart from real
    samples, hence this check.

    This is an importance-sampling efficiency problem, not a statistical one.
    The Gamma stage is the exact flat-prior posterior and is calibrated with
    respect to that prior. But the truths come from the astrophysical population,
    and a flat prior on a distance favours large dL far more than the population
    does, so the truth sits low in its own posterior -- at k=3 it is below the
    5th percentile for ~16% of events. The SNR term applied here is much sharper
    than the dL "measurement" and pulls back toward the truth, i.e. into that
    lower tail, so only a few proposals carry weight. Smaller concentrations
    widen the proposal and make the overlap worse.
    """
    ess = np.asarray(1.0 / jnp.sum(p**2, axis=1))
    low = ess < num_posterior_samples
    if not low.any():
        return ess

    worst = int(np.argmin(ess))
    print(
        f"Warning: SNR reweighting effective sample size is below the requested "
        f"{num_posterior_samples} samples for {int(low.sum())} of {ess.shape[0]} "
        f"events in this batch (worst: event {worst + event_offset}, ESS "
        f"{ess[worst]:.1f}). Those events will contain duplicated samples. "
        f"Raise the concentrations so the Gamma proposal overlaps the "
        f"SNR-reweighted target better, or raise buffer_factor."
    )
    return ess


def reweigh_samples_with_snr_contribution(
    samples,
    samples_true,
    num_posterior_samples,
    key,
    luminosity_distance_likelihood,
    log_weight_extra=None,
):
    """Draw luminosity_distance from the SNR constraint and importance-correct.

    The obvious scheme -- propose dL from its own Gamma posterior, then reweight
    by the chi2 SNR likelihood -- targets a thin sheet: at fixed iota the SNR
    pins dL to a few percent, while iota smears it over the 2.8x range of the
    inclination factor. Independent (dL, iota) draws almost never land on that
    sheet, so the effective sample size collapses (0.2% of proposals, and far
    worse for events whose truth sits in the tail of the dL posterior).

    Instead draw the chi2 residual directly and solve for dL:

        s ~ chi2(df=2),  snr = sqrt(snr_obs**2 - s),  dL = C(Mc, q, iota) / snr

    The chi2 factor then cancels against the proposal density and the weight
    collapses to the (broad) dL measurement likelihood times the change of
    variables |ds/ddL| = 2 C**2 / dL**3:

        log w = log L_dL(dL) + 3 log(dL) - log(2 C**2)

    Same target posterior, ~250-4000x the effective sample size. dL is bounded
    below by C / snr_obs, so the weight cannot diverge.
    """

    key_iota, key_residual, key_resample = jax.random.split(_as_key(key), 3)

    num_events, num_proposals = samples.chirp_mass_d.shape
    iota_samples = get_iota_prior_samples(
        num_events=num_events,
        num_posterior_samples=num_proposals,
        key=key_iota,
    )

    coefficient = snr_coefficient(
        samples.chirp_mass_d,
        samples.mass_ratio,
        iota_samples,
        samples_true.config.amplitude,
    )

    snr_obs_squared = samples_true.scatter_snr[:, None] ** 2
    # chi2(df=2) is Exponential with scale 2
    residual = 2.0 * jax.random.exponential(key_residual, (num_events, num_proposals))
    valid = residual < snr_obs_squared
    snr = jnp.sqrt(jnp.where(valid, snr_obs_squared - residual, snr_obs_squared))
    luminosity_distance = coefficient / snr

    # concentration may be scalar or per-event; broadcast so both work
    shape_luminosity_distance = jnp.broadcast_to(
        jnp.asarray(luminosity_distance_likelihood.concentration) + 1.0,
        jnp.shape(luminosity_distance_likelihood.data),
    )[:, None]
    log_prob_luminosity_distance = numpyro.distributions.Gamma(
        shape_luminosity_distance,
        luminosity_distance_likelihood.data[:, None],
    ).log_prob(luminosity_distance)

    log_p = jnp.where(
        valid,
        log_prob_luminosity_distance
        + 3.0 * jnp.log(luminosity_distance)
        - jnp.log(2.0 * coefficient**2),
        -jnp.inf,
    )

    if log_weight_extra is not None:
        # importance weights from the other parameters' proposals (e.g. the
        # mass-ratio truncation) fold into the same single resampling step
        log_p = log_p + log_weight_extra
        valid = valid & jnp.isfinite(log_weight_extra)

    empty = ~np.asarray(jnp.any(valid, axis=1))
    if empty.any():
        raise ValueError(
            f"No usable proposal for events {np.where(empty)[0].tolist()}. "
            "Raise buffer_factor or check the SNR model."
        )

    samples.luminosity_distance = luminosity_distance

    p = jnp.exp(log_p - jax.scipy.special.logsumexp(log_p, axis=1, keepdims=True))

    warn_on_low_reweighting_ess(p, num_posterior_samples)

    idx = resample_idx_from_p(p, p.shape[0], num_posterior_samples, key_resample)

    samples_f = SimpleNamespace()
    for attr in [
        "chirp_mass_d",
        "mass_ratio",
        "luminosity_distance",
        "prior_masses_d_dL",
    ]:
        if not hasattr(samples, attr):
            continue
        v = getattr(samples, attr)

        rows = jnp.arange(v.shape[0])[:, None]
        setattr(samples_f, attr, v[rows, idx])

    samples_f.mass_1_d = get_mass_1_from_chirp_mass_and_mass_ratio(
        samples_f.chirp_mass_d, samples_f.mass_ratio
    )

    return samples_f


def draw_gamma_observation(rate_true, concentration, key):
    """Mock observation of a positive quantity: ``d | x ~ Gamma(k, rate=x)``.

    With a flat prior on x the posterior is ``Gamma(k + 1, rate=d)``.
    """
    return (
        numpyro.distributions.Gamma(concentration, rate_true)
        .sample(_as_key(key), (1,))
        .squeeze()
    )


def snr_coefficient(chirp_mass_d, mass_ratio, iota, amplitude):
    """C in ``snr = C / luminosity_distance``, i.e. the SNR at unit distance.

    ``optimal_snr_approximated`` depends on the distance only through a final
    ``/ distance``, so evaluating it at distance = 1 gives C exactly. Deferring
    to it rather than restating the formula keeps the mock parameter estimation
    and the catalog generation from drifting apart.
    """
    return optimal_snr_approximated(
        chirp_mass_d=chirp_mass_d,
        mass_ratio=mass_ratio,
        distance=1.0,
        iota=iota,
        amplitude=amplitude,
    )


def get_unit_interval_samples_from_gamma_likelihood(
    rate_true, concentration, num_samples, key
):
    """Proposal and log importance weight for a Gamma-likelihood parameter in (0, 1).

    Target: ``Gamma(concentration + 1, rate=data)`` truncated to (0, 1) -- the
    flat-prior posterior for ``data | q ~ Gamma(concentration, rate=q)``.

    Proposing from the untruncated posterior and rejecting is very wasteful here.
    The posterior mean sits at ``(k+1)/(k-1)`` times q_true, which for the typical
    mass ratio is above 1, so the truncation throws away ~56% of proposals on
    average and >99% for some events. Instead the proposal rate is tilted so the
    proposal mean matches the *truncated* target mean -- available in closed form,
    ``E[q | q<1] = (a/rate) P(a+1, rate) / P(a, rate)`` -- and the mismatch is
    corrected by an importance weight rather than by discarding samples.

    Returns ``(samples, log_weight)``. The weight is -inf above q = 1, so the
    truncation is enforced by the weight and no separate rejection stage is needed.
    """
    key_data, key_proposal = jax.random.split(_as_key(key))

    data = draw_gamma_observation(rate_true, concentration, key_data)
    # concentration may be a scalar or one value per event; make it per-event so
    # the (num_events, num_samples) expressions below broadcast either way
    shape = jnp.broadcast_to(jnp.asarray(concentration) + 1.0, jnp.shape(data))

    floor = jnp.finfo(jnp.asarray(data).dtype).tiny
    prob_shape = jax.scipy.special.gammainc(shape, data)
    prob_shape_plus = jax.scipy.special.gammainc(shape + 1.0, data)
    truncated_mean = (shape / data) * prob_shape_plus / jnp.maximum(prob_shape, floor)
    tilt = (shape / data) / jnp.maximum(truncated_mean, floor)
    rate_proposal = tilt * data

    samples = (
        numpyro.distributions.Gamma(shape, rate_proposal)
        .sample(key_proposal, (num_samples,))
        .T
    )

    # log[Gamma(shape, data) / Gamma(shape, rate_proposal)] on (0, 1), else -inf
    log_weight = jnp.where(
        samples < 1.0,
        -shape[:, None] * jnp.log(tilt)[:, None]
        + (rate_proposal - data)[:, None] * samples,
        -jnp.inf,
    )

    return samples, log_weight


def get_samples_from_gamma_likelihood(rate_true, config, key):

    concentration = config.concentration
    prior_low, prior_high = config.prior_low, config.prior_high
    num_posterior_samples = config.num_posterior_samples

    # separate streams for the mock observation, the posterior draw and the
    # truncation resampling; previously the first and third both used
    # PRNGKey(seed) and the second used PRNGKey(seed + 1)
    key_data, key_post, key_resample = jax.random.split(_as_key(key), 3)

    # need to draw more samples if we are cutting on them
    if prior_low is not None and prior_high is not None:
        cut_domain = True
    elif prior_low is None and prior_high is None:
        cut_domain = False
    else:
        raise NotImplementedError("Cutting on only one side not implemented")

    data = (
        numpyro.distributions.Gamma(concentration, rate_true)
        .sample(key_data, (1,))
        .squeeze()
    )

    # draw posterior samples for each data point
    rate_samples_outside_prior = (
        numpyro.distributions.Gamma(concentration + 1, data)
        .sample(key_post, (num_posterior_samples,))
        .T
    )

    if cut_domain:
        rate_samples = resample_on_finite_domain(
            rate_samples_outside_prior,
            num_posterior_samples,
            prior_low,
            prior_high,
            key_resample,
        )
    else:
        rate_samples = rate_samples_outside_prior

    return rate_samples


def resample_on_finite_domain(
    samples, num_posterior_samples, prior_low, prior_high, key
):
    """
    Resample samples to be within [prior_low, prior_high]
    using rejection sampling.

    samples: jnp.ndarray of shape (num_events, num_posterior_samples)
    prior_low: float
    prior_high: float

    """
    num_events, _ = samples.shape

    p = (samples >= prior_low) & (samples <= prior_high)
    p = p / jnp.sum(p, axis=1, keepdims=True)

    idx = resample_idx_from_p(p, num_events, num_posterior_samples, key)

    rows = jnp.arange(samples.shape[0])[:, None]

    return samples[rows, idx]


def resample_idx_from_p(p, num_events, num_posterior_samples, key):

    def get_samples_for_distribution(key, p_dist):
        return jax.random.choice(
            key, p.shape[-1], shape=(num_posterior_samples,), p=p_dist, replace=True
        )

    keys = jax.random.split(_as_key(key), num_events)
    idx = jax.vmap(get_samples_for_distribution, in_axes=(0, 0))(keys, p)

    return idx
