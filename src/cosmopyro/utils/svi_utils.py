from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers
from numpyro.infer import autoguide

from .jax_utils import group_sizes_are_uniform, make_segment_ids_from_group_sizes

__all__ = [
    "build_guide",
    "get_not_dense_params",
    "inverse_mass_matrix_from_guide_samples",
    "inverse_mass_matrix_from_hessian",
    "subset_data",
]


def subset_data(data, num_events):
    """Return a copy of data with only the first num_events events."""
    n = data.samples.num_posterior_samples_per_event
    total_events = n.shape[0]
    if num_events >= total_events:
        return data

    n_sub = n[:num_events]
    n_flat = int(jnp.sum(n_sub))
    groups_are_uniform = group_sizes_are_uniform(n_sub)

    sub_samples = SimpleNamespace()
    for attr, v in vars(data.samples).items():
        if attr == "num_posterior_samples_per_event":
            setattr(sub_samples, attr, n_sub)
        elif attr == "posterior_sample_groups_are_uniform":
            setattr(sub_samples, attr, groups_are_uniform)
        elif attr == "posterior_sample_segment_ids":
            if not groups_are_uniform:
                setattr(sub_samples, attr, make_segment_ids_from_group_sizes(n_sub))
        elif hasattr(v, "shape") and v.ndim == 1 and v.shape[0] == int(jnp.sum(n)):
            setattr(sub_samples, attr, v[:n_flat])
        else:
            setattr(sub_samples, attr, v)

    if getattr(sub_samples, "posterior_sample_groups_are_uniform", None) is None:
        sub_samples.posterior_sample_groups_are_uniform = groups_are_uniform

    if (
        not sub_samples.posterior_sample_groups_are_uniform
        and getattr(sub_samples, "posterior_sample_segment_ids", None) is None
    ):
        sub_samples.posterior_sample_segment_ids = make_segment_ids_from_group_sizes(
            n_sub
        )

    sub_data = SimpleNamespace()
    sub_data.samples = sub_samples
    sub_data.injections = data.injections
    return sub_data


def get_not_dense_params(kwargs_sampler):
    """Return the list of non-dense (high-dimensional) parameter names."""
    return kwargs_sampler.get("not_dense_params", ["gaussian_F_whitened_spatial_white"])


def inverse_mass_matrix_from_hessian(
    model,
    model_args,
    model_kwargs,
    mode,
    dense_groups,
    diag_sites,
    floor=1e-8,
    eig_floor=1.0,
):
    """Build NUTS ``inverse_mass_matrix`` as a prior-floored Laplace approximation.

    Posterior precision = likelihood precision + prior precision ≥ prior
    precision. So when the data Hessian gives a tiny or negative eigenvalue
    along some direction (weakly identified parameter, or a saddle), the right
    answer isn't "huge variance there" — it's "fall back to the prior."

    Implementation: compute H = ∇² (-log posterior) at ``mode`` in NumPyro's
    unconstrained sample space, eigendecompose, then floor any eigenvalue
    below ``eig_floor`` to ``eig_floor`` before inverting. The default
    ``eig_floor=1.0`` matches the prior precision used by both reparam
    families in CosmoPyro:
      - ``_white`` sites (Normal whitened):    Normal(0,1) precision = 1.
      - ``_base`` sites (Uniform via sigmoid): unconstrained Jacobian gives
        precision ≈ 0.5–1 in the typical region, ~1 is a sane floor.

    With this floor, weakly-constrained dims get ``inverse_mass = 1`` (the
    prior covariance), and tightly-constrained dims keep their data-driven
    smaller variance. NUTS' step-size budget is then set by the actual
    likelihood-driven curvature, not by saddle artifacts.

    Computed in NumPyro's *unconstrained* parameter space (the same space NUTS
    samples in), so reparameterised Uniform priors get the sigmoid bijector
    automatically.

    Parameters
    ----------
    model, model_args, model_kwargs : NumPyro model + args/kwargs.
    mode : dict[str, jnp.ndarray]
        Constrained sample-site values to expand around (typically the SVI
        ``best_sample``). Need not be a true posterior mode — the prior floor
        protects against saddle artifacts.
    dense_groups : list[tuple[str, ...]]
        Must match the ``dense_mass`` argument of NUTS verbatim.
    diag_sites : list[str]
        Sites with diagonal mass.
    floor : float
        Minimum variance on diagonals so the matrix stays invertible.
    eig_floor : float
        Hessian eigenvalues below this are clipped to it. Default 1.0 = prior
        precision. Set lower (e.g. 1e-3) only if you trust your mode is a real
        posterior peak and you want the saddle directions to be unconstrained
        in the mass matrix.

    Returns
    -------
    dict[tuple[str, ...], jnp.ndarray] : NumPyro-compatible inverse mass matrix.
    """
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model

    # Move into NumPyro's unconstrained parameter space (so reparameterised
    # Uniform priors are mapped via sigmoid → ℝ and there are no boundaries).
    rng_key = jax.random.key(0)
    init_info = initialize_model(
        rng_key,
        model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        init_strategy=init_to_value(values=mode),
    )
    z = init_info[0].z  # unconstrained dict of sample sites
    potential_fn = init_info[1]  # callable taking unconstrained dict
    flat_z, unflatten = _ravel(z)

    def pot_flat(x):
        return potential_fn(unflatten(x))

    # Hessian in unconstrained space.
    H = jax.hessian(pot_flat)(flat_z)
    H = 0.5 * (H + H.T)

    # Project to PSD: clip small/negative eigenvalues. SVI mode often parks
    # near a saddle (negative eig along uninformed directions), so this is
    # important for robustness.
    eigs, vecs = jnp.linalg.eigh(H)
    eigs = jnp.where(eigs < eig_floor, eig_floor, eigs)
    H_inv = (vecs * (1.0 / eigs)) @ vecs.T

    # Map flat indices → site name (preserves the order produced by ravel_pytree).
    site_indices = {}
    cursor = 0
    for k, v in z.items():
        n = int(jnp.size(v))
        site_indices[k] = list(range(cursor, cursor + n))
        cursor += n

    imm = {}

    for group in dense_groups:
        idxs = []
        missing = []
        for s in group:
            if s in site_indices:
                idxs.extend(site_indices[s])
            else:
                missing.append(s)
        if missing:
            print(
                f"[hessian_mass] dense group {group}: sites missing from mode: {missing}"
            )
        if not idxs:
            continue
        sub = H_inv[jnp.array(idxs)[:, None], jnp.array(idxs)[None, :]]
        d = jnp.diag(sub)
        sub = sub + jnp.diag(jnp.maximum(d, floor) - d)
        imm[group] = sub

    for s in diag_sites:
        if s not in site_indices:
            print(f"[hessian_mass] diag site '{s}' missing from mode — skipping")
            continue
        idxs = jnp.array(site_indices[s])
        var = jnp.diag(H_inv)[idxs]
        imm[(s,)] = jnp.maximum(var, floor)

    return imm


def _ravel(d):
    import jax.flatten_util as ju

    return ju.ravel_pytree(d)


def inverse_mass_matrix_from_guide_samples(
    posterior_samples, dense_groups, diag_sites, site_sizes=None, floor=1e-8
):
    """Build a NUTS ``inverse_mass_matrix`` dict from SVI guide samples.

    The inverse mass matrix in HMC equals the posterior covariance. SVI's
    guide gives an estimate of that covariance for free — much better than
    letting NUTS rediscover it from identity during a short warmup.

    Parameters
    ----------
    posterior_samples : dict[str, jnp.ndarray]
        Samples from ``Predictive(guide)``. Each value has shape
        ``(num_chains, num_samples, *site_shape)`` or
        ``(num_samples, *site_shape)``.
    dense_groups : list[tuple[str, ...]]
        Must exactly match the ``dense_mass`` argument of ``NUTS`` (each tuple
        is one dense block). The returned dict uses these tuples verbatim as
        keys so NumPyro's _initialize_mass_matrix can look them up.
    diag_sites : list[str]
        Sites with diagonal mass (each becomes its own 1-tuple key).
    site_sizes : dict[str, int] or None
        Flattened element count per site, used to fall back to prior variance
        (=1) for any site missing from ``posterior_samples``. If omitted,
        missing sites are skipped and NUTS will re-initialize them.
    floor : float
        Minimum allowed variance so the matrix stays invertible even if SVI
        collapsed a site to a delta.

    Returns
    -------
    dict[tuple[str, ...], jnp.ndarray]
        Mapping accepted by ``NUTS(..., inverse_mass_matrix=...)``. Dense
        groups map to 2D cov matrices; diag sites map to 1D variance vectors.
    """
    import jax

    imm = {}

    def _squeeze_chains(v):
        # Accept (samples, *shape) or (chains, samples, *shape); in the latter
        # case merge the leading two axes into one "sample" axis.
        return v.reshape((-1,) + v.shape[2:]) if v.ndim >= 2 and v.shape[0] == 1 else v

    # pick a sample count from any guide sample (used for prior fill-ins)
    any_site = next(iter(posterior_samples.values()))
    n_fake = int(_squeeze_chains(any_site).shape[0])

    def _flat_samples_for(site):
        """Return (n, size) flattened samples for a site, or None if missing."""
        if site in posterior_samples:
            v = _squeeze_chains(posterior_samples[site])
            return v.reshape(v.shape[0], -1)
        return None

    def _prior_flat(site):
        """Standard-normal fallback (N(0,1) prior) for a missing site."""
        size = (site_sizes or {}).get(site, 1)
        return jax.random.normal(
            jax.random.key(abs(hash(site)) % (2**31)), (n_fake, size)
        )

    # --- Dense blocks: one joint covariance per group tuple ---
    for group in dense_groups:
        flats = []
        missing = []
        for site in group:
            f = _flat_samples_for(site)
            if f is None:
                missing.append(site)
                f = _prior_flat(site)
            flats.append(f)
        if missing:
            print(
                f"[inverse_mass] dense group {group}: missing from guide, using prior N(0,1): {missing}"
            )
        combined = jnp.concatenate(flats, axis=-1)  # (n_samples, total_size)
        if combined.shape[-1] == 1:
            cov = jnp.array([[max(float(jnp.var(combined)), floor)]])
        else:
            cov = jnp.cov(combined.T)
            diag = jnp.diag(cov)
            cov = cov + jnp.diag(jnp.clip(diag, floor) - diag)  # floor the diag
        imm[group] = cov  # key MUST match the dense_mass tuple verbatim

    # --- Diagonal sites: per-site 1-tuple key ---
    for s in diag_sites:
        f = _flat_samples_for(s)
        if f is None:
            size = (site_sizes or {}).get(s, 1)
            print(
                f"[inverse_mass] diag site '{s}' missing — using prior variance 1.0 (size {size})"
            )
            imm[(s,)] = jnp.ones(size)
            continue
        var = jnp.clip(jnp.var(f, axis=0), floor)
        imm[(s,)] = var

    return imm


def _related_site_names(not_dense_params):
    """Return all message names tied to the GP sample sites.

    For a sample site ``foo_white`` created by ``get_Normal_samples``,
    the model also produces:
      - the deterministic ``foo``
      - the plate frames ``foo_plate_0``, ``foo_plate_1``, ...

    Returns the GP sample names + their deterministic counterparts, and
    a tuple of plate-name prefixes for substring matching.
    """
    site_names = set(not_dense_params)
    plate_prefixes = []
    for p in not_dense_params:
        base = p.removesuffix("_white")
        site_names.add(base)
        plate_prefixes.append(f"{base}_plate_")
    return site_names, tuple(plate_prefixes)


def _make_hide_fn(not_dense_params, hide_dense):
    """Build a hide_fn for ``numpyro.handlers.block``.

    Always hides deterministic sites (autoguides in ``AutoGuideList`` reject
    them).  In addition:

    - ``hide_dense=True``: hide everything except the GP sample sites and
      their plate frames (used for the GP guide).
    - ``hide_dense=False``: hide only the GP sample sites and their plates
      (used for the dense guide).
    """
    gp_site_names, plate_prefixes = _related_site_names(not_dense_params)

    def is_gp_related(msg):
        name = msg.get("name", "")
        if msg["type"] == "plate":
            return name.startswith(plate_prefixes)
        return name in gp_site_names

    def hide_fn(msg):
        # Deterministics can't be in sub-guide prototype traces of AutoGuideList.
        if msg["type"] == "deterministic":
            return True
        if hide_dense:
            return not is_gp_related(msg)
        return is_gp_related(msg)

    return hide_fn


def build_guide(model, kwargs_sampler):
    """Construct an SVI guide from sampler settings.

    Supported guide_type values:
        - 'AutoLowRankMultivariateNormal'
        - 'AutoIAFNormal'
        - 'AutoBNAFNormal'
        - 'AutoBNAFNormal_AutoNormal'  — AutoBNAFNormal for dense params,
          AutoNormal for non-dense (high-dimensional) params like the GP field.

    Returns
    -------
    guide : AutoGuide
    """
    guide_type = kwargs_sampler.get("guide_type", "AutoLowRankMultivariateNormal")

    if guide_type == "AutoLowRankMultivariateNormal":
        return autoguide.AutoLowRankMultivariateNormal(model)

    elif guide_type == "AutoIAFNormal":
        hidden_dims = kwargs_sampler.get("guide_hidden_dims")
        if hidden_dims is None:
            return autoguide.AutoIAFNormal(model)
        return autoguide.AutoIAFNormal(model, hidden_dims=hidden_dims)

    elif guide_type == "AutoBNAFNormal":
        hidden_factors = kwargs_sampler.get("guide_hidden_factors", [10, 10, 10])
        return autoguide.AutoBNAFNormal(model, hidden_factors=hidden_factors)

    elif guide_type == "AutoBNAFNormal_AutoNormal":
        not_dense_params = get_not_dense_params(kwargs_sampler)
        hidden_factors = kwargs_sampler.get("guide_hidden_factors", [10, 10, 10])

        guide = autoguide.AutoGuideList(model)
        # BNAF for the dense (scalar) parameters; hide GP sites + their plates
        guide.append(
            autoguide.AutoBNAFNormal(
                numpyro.handlers.block(
                    model, hide_fn=_make_hide_fn(not_dense_params, hide_dense=False)
                ),
                hidden_factors=hidden_factors,
            )
        )
        # Diagonal normal for the GP field; hide everything else but keep GP plates
        guide.append(
            autoguide.AutoNormal(
                numpyro.handlers.block(
                    model, hide_fn=_make_hide_fn(not_dense_params, hide_dense=True)
                ),
            )
        )
        return guide

    elif guide_type == "AutoLowRankMVN_AutoNormal":
        not_dense_params = get_not_dense_params(kwargs_sampler)
        rank = kwargs_sampler.get("guide_rank", None)

        guide = autoguide.AutoGuideList(model)
        # Low-rank MVN for the dense (scalar) parameters — captures correlations
        guide.append(
            autoguide.AutoLowRankMultivariateNormal(
                numpyro.handlers.block(
                    model, hide_fn=_make_hide_fn(not_dense_params, hide_dense=False)
                ),
                **(dict(rank=rank) if rank is not None else {}),
            )
        )
        # Diagonal normal for the GP field; hide everything else but keep GP plates
        guide.append(
            autoguide.AutoNormal(
                numpyro.handlers.block(
                    model, hide_fn=_make_hide_fn(not_dense_params, hide_dense=True)
                ),
            )
        )
        return guide

    else:
        raise ValueError(
            f"Unknown guide_type: '{guide_type}'. Choose from: "
            "AutoLowRankMultivariateNormal, AutoIAFNormal, AutoBNAFNormal, "
            "AutoBNAFNormal_AutoNormal, AutoLowRankMVN_AutoNormal"
        )
