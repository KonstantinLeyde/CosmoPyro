from types import SimpleNamespace

import jax
import jax.numpy as jnp

from ..cosmology import cosmology
from ..distributions import mass_distribution_parametrized
from ..distributions.redshift import MadauDickinsonRedshiftModel
from ..utils.jax_utils import compute_centers_and_delta_from_array
from .data_utils import save_namespace_to_hdf5

__all__ = [
    "concatenate_namespaces",
    "draw_masses_and_snr",
    "generate_catalog",
    "generate_catalog_from_galaxy_catalog",
    "optimal_snr_approximated",
    "sample_host_galaxies",
]


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------


def optimal_snr_approximated(chirp_mass_d, mass_ratio, distance, iota, amplitude):
    """Approximated GW SNR including bandwidth penalty for high total mass."""
    eta = mass_ratio / ((1 + mass_ratio) ** 2)
    total_mass = chirp_mass_d / (eta ** (3 / 5))
    inc = jnp.sqrt(((1 + jnp.cos(iota) ** 2) ** 2) / 4.0 + jnp.cos(iota) ** 2)

    m_cutoff = 200.0
    bandwidth_penalty = 1.0 / (1.0 + (total_mass / m_cutoff) ** 2)

    return amplitude * (chirp_mass_d ** (5 / 6)) * inc * bandwidth_penalty / distance


# ---------------------------------------------------------------------------
# Grid sampling helpers
# ---------------------------------------------------------------------------


def _sample_from_arrays(x, log_prob, num_samples, key):
    prob = jnp.exp(log_prob - jax.scipy.special.logsumexp(log_prob))
    idx = jax.random.choice(key, a=x.shape[0], shape=(num_samples,), p=prob)
    return x[idx], log_prob[idx]


def _sample_from_2d_arrays(x, y, log_prob, num_samples, key):
    prob = jnp.exp(log_prob - jax.scipy.special.logsumexp(log_prob))
    flat_idx = jax.random.choice(
        key, a=prob.size, shape=(num_samples,), p=prob.flatten()
    )
    xi = flat_idx // y.shape[0]
    yi = flat_idx % y.shape[0]
    return x[xi], y[yi], log_prob.flatten()[flat_idx]


# ---------------------------------------------------------------------------
# Shared: draw masses, detector-frame quantities, and SNR
# ---------------------------------------------------------------------------


def draw_masses_and_snr(config, params, z_samples, cosmo, key):
    """Draw masses from grid, compute detector-frame quantities and SNR."""
    bin_cfg = config["binning"]
    key_m, key_incl, key_scatter = jax.random.split(key, 3)

    m1_edges = jnp.exp(
        jnp.linspace(
            jnp.log(bin_cfg["m1_min"]), jnp.log(bin_cfg["m1_max"]), bin_cfg["m1_nbins"]
        )
    )
    q_edges = jnp.linspace(bin_cfg["q_min"], bin_cfg["q_max"], bin_cfg["q_nbins"])

    m1_centers = compute_centers_and_delta_from_array(m1_edges)[0]
    q_centers = compute_centers_and_delta_from_array(q_edges)[0]

    analysis = SimpleNamespace(
        binning=dict(
            boundaries=dict(mass_1_s=m1_edges, mass_ratio=q_edges),
            centers=dict(mass_1_s=m1_centers, mass_ratio=q_centers),
        )
    )

    prob_m1s = mass_distribution_parametrized.construct_prob_nn_multipeak_1D(
        analysis, params
    )
    prob_q = (
        mass_distribution_parametrized.construct_running_power_law_prob_mass_ratio_nn(
            analysis, params, m1_centers, q_centers
        )
    )
    # add jacobian since we have a log-space grid in m1
    jacobian = m1_centers[:, None]
    prob_mass = prob_m1s[:, None] * jacobian * prob_q.T

    num_samples = z_samples.shape[0]
    m1s, q, log_pm = _sample_from_2d_arrays(
        m1_centers, q_centers, jnp.log(prob_mass), num_samples, key_m
    )

    # undo jacobian for log_prob_mass
    log_pm = log_pm - jnp.log(m1s)

    # Derived quantities
    m2s = m1s * q
    chirp_s = (m1s * m2s) ** (3 / 5) / (m1s + m2s) ** (1 / 5)
    one_plus_z = 1 + z_samples
    m1d = one_plus_z * m1s
    m2d = one_plus_z * m2s
    chirp_d = one_plus_z * chirp_s
    dL = cosmo.get_luminosity_distance_gw_from_redshift(z_samples)

    # Inclination and SNR
    cos_iota = jax.random.uniform(
        key_incl, shape=(num_samples,), minval=-1.0, maxval=1.0
    )
    iota = jnp.arccos(cos_iota)
    snr_opt = optimal_snr_approximated(chirp_d, q, dL, iota, config["amplitude"])
    scatter = jax.random.chisquare(key_scatter, df=2, shape=(num_samples,))
    scatter_snr = jnp.sqrt(snr_opt**2 + scatter)

    return SimpleNamespace(
        mass_1_s=m1s,
        mass_2_s=m2s,
        mass_ratio=q,
        chirp_mass_s=chirp_s,
        mass_1_d=m1d,
        mass_2_d=m2d,
        chirp_mass_d=chirp_d,
        luminosity_distance=dL,
        cos_iota=cos_iota,
        iota=iota,
        optimal_snr=snr_opt,
        scatter_snr=scatter_snr,
        log_prob_mass=log_pm,
    )


# ---------------------------------------------------------------------------
# Entry point 1: smooth Madau-Dickinson redshift sampling
# ---------------------------------------------------------------------------


def generate_catalog(config, out_path):
    """
    Generate an injection catalog by sampling redshifts from a Madau-Dickinson
    model and masses from a multipeak power-law model.

    Parameters
    ----------
    config : dict
        num_samples, seed, snr_threshold, amplitude,
        cosmology: {H0, Omega_m},
        redshift: {gamma, kappa, zp},
        mass_model: {mass_1_s: {...}, mass_ratio: {...}},
        binning: {m1_min, m1_max, m1_nbins, q_min, q_max, q_nbins, nz_grid}.
    out_path : str
        HDF5 output path.
    """
    num_samples = int(config["num_samples"])
    seed = int(config["seed"])
    snr_thr = float(config["snr_threshold"])
    amplitude = float(config["amplitude"])

    cosmo_cfg = config["cosmology"]
    H0 = float(cosmo_cfg["H0"])
    Omega_m = float(cosmo_cfg["Omega_m"])

    rz_cfg = config["redshift"]
    rz_params = dict(
        redshift=dict(
            gamma=float(rz_cfg["gamma"]),
            kappa=float(rz_cfg["kappa"]),
            zp=float(rz_cfg["zp"]),
        ),
        cosmology=dict(H0=H0, Omega_m=Omega_m),
    )

    params = config["mass_model"]
    nz_grid = int(config["binning"]["nz_grid"])

    # RNG
    key = jax.random.PRNGKey(seed)
    key_z, key_mass = jax.random.split(key)

    # Cosmology
    cosmo_name = config.get("cosmology_name", "FlatLambdaCDM")
    params_ratio = config.get("params_ratio", {}) or {}

    if cosmo_name == "FlatLambdaCDM":
        cosmo = cosmology.FlatLambdaCDM(params=dict(cosmo_cfg))
    elif cosmo_name in [
        "FlatLambdaCDM_GW_distance_cosine",
        "FlatLambdaCDM_GW_distance_gp_integrated",
        "FlatLambdaCDM_GW_distance_cM",
    ]:
        cosmo = cosmology.ModifiedGWDistanceFlatLambdaCDM(
            params=dict(cosmo_cfg),
            name_modified_ratio=cosmo_name,
            params_modified_gravity=params_ratio if params_ratio else None,
        )
    else:
        raise NotImplementedError(f"Unknown cosmology_name: {cosmo_name}")

    # Sample redshifts from Madau-Dickinson
    rz_model = MadauDickinsonRedshiftModel(cosmo, rz_params)
    zmin = float(jnp.min(cosmo.z_interp) + 1e-4)
    zmax = float(jnp.max(cosmo.z_interp))
    z_vals = jnp.linspace(zmin, zmax, nz_grid)
    log_pz = rz_model.log_prob(dict(redshift=z_vals))
    z_samples, log_pz_draw = _sample_from_arrays(z_vals, log_pz, num_samples, key_z)

    # Draw masses and SNR
    draws = draw_masses_and_snr(config, params, z_samples, cosmo, key_mass)

    # Selection
    sel = draws.scatter_snr >= snr_thr

    # Jacobian: (z, m1_s, q) -> (m1_d, m2_d, dL)
    log_p_draw = log_pz_draw[sel] + draws.log_prob_mass[sel]
    dDdz = cosmo.get_dluminosity_distance_gw_over_dz_from_redshift(z_samples)
    jac = dDdz * draws.mass_1_d * (1 + z_samples)
    log_p_m1d_m2d_dL = log_p_draw - jnp.log(jac[sel])

    cfg_ns = SimpleNamespace(
        seed=seed,
        snr_threshold=snr_thr,
        amplitude=amplitude,
        cosmology=dict(H0=H0, Omega_m=Omega_m),
        redshift=rz_params["redshift"],
        mass_model=params,
    )

    cat = SimpleNamespace(
        redshift=z_samples[sel],
        log_prob_redshift=log_pz_draw[sel],
        mass_1_s=draws.mass_1_s[sel],
        mass_2_s=draws.mass_2_s[sel],
        mass_ratio=draws.mass_ratio[sel],
        chirp_mass_s=draws.chirp_mass_s[sel],
        log_prob_mass_1_s_mass_ratio=draws.log_prob_mass[sel],
        mass_1_d=draws.mass_1_d[sel],
        mass_2_d=draws.mass_2_d[sel],
        chirp_mass_d=draws.chirp_mass_d[sel],
        luminosity_distance=draws.luminosity_distance[sel],
        cos_iota=draws.cos_iota[sel],
        iota=draws.iota[sel],
        optimal_snr=draws.optimal_snr[sel],
        scatter_snr=draws.scatter_snr[sel],
        log_prob_draw_redshift_mass_1_s_mass_ratio=log_p_draw,
        log_prob_mass_1_d_mass_2_d_luminosity_distance=log_p_m1d_m2d_dL,
        num_total_injections=jnp.array([num_samples]),
        snr_threshold=jnp.array([snr_thr]),
        seed=jnp.array([seed]),
        config=cfg_ns,
    )

    save_namespace_to_hdf5(cat, out_path)
    return cat


# ---------------------------------------------------------------------------
# Entry point 2: galaxy catalog host sampling
# ---------------------------------------------------------------------------


def sample_host_galaxies(galaxy_catalog, num_samples, key):
    """Sample galaxy indices weighted by host_log_prob and time dilation."""
    z = galaxy_catalog.redshift
    log_weights = galaxy_catalog.host_log_prob - jnp.log(1 + z)
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    prob = jnp.exp(log_weights)

    idx = jax.random.choice(key, a=prob.shape[0], shape=(num_samples,), p=prob)
    return idx, log_weights[idx]


def generate_catalog_from_galaxy_catalog(config, galaxy_catalog, out_path=None):
    """
    Generate a GW event catalog by drawing host galaxies from a galaxy catalog.

    The galaxy catalog's host_log_prob is assumed to already encode the
    comoving volume and star formation rate; only time dilation is added.

    Parameters
    ----------
    config : dict
        num_samples, seed, snr_threshold, amplitude,
        cosmology: {H0, Omega_m},
        mass_model: {mass_1_s: {...}, mass_ratio: {...}},
        binning: {m1_min, m1_max, m1_nbins, q_min, q_max, q_nbins}.
    galaxy_catalog : SimpleNamespace
        Must have .redshift, .ra, .dec, .host_log_prob arrays.
    out_path : str, optional
        If given, save catalog to this HDF5 path.
    """
    num_samples = int(config["num_samples"])
    seed = int(config["seed"])
    snr_thr = float(config["snr_threshold"])

    cosmo_cfg = config["cosmology"]
    cosmo = cosmology.FlatLambdaCDM(params=dict(cosmo_cfg))
    params = config["mass_model"]

    key = jax.random.PRNGKey(seed)
    key_gal, key_mass = jax.random.split(key)

    gal_idx, log_weight_draw = sample_host_galaxies(
        galaxy_catalog, num_samples, key_gal
    )
    z_samples = galaxy_catalog.redshift[gal_idx]

    draws = draw_masses_and_snr(config, params, z_samples, cosmo, key_mass)

    sel = draws.scatter_snr >= snr_thr

    dDdz = cosmo.get_dluminosity_distance_gw_over_dz_from_redshift(z_samples)
    jac = dDdz * draws.mass_1_d * (1 + z_samples)
    log_p_draw = log_weight_draw + draws.log_prob_mass
    log_p_m1d_m2d_dL = log_p_draw[sel] - jnp.log(jac[sel])

    cat = SimpleNamespace(
        galaxy_index=gal_idx[sel],
        redshift=z_samples[sel],
        mass_1_s=draws.mass_1_s[sel],
        mass_2_s=draws.mass_2_s[sel],
        mass_ratio=draws.mass_ratio[sel],
        chirp_mass_s=draws.chirp_mass_s[sel],
        mass_1_d=draws.mass_1_d[sel],
        mass_2_d=draws.mass_2_d[sel],
        chirp_mass_d=draws.chirp_mass_d[sel],
        luminosity_distance=draws.luminosity_distance[sel],
        cos_iota=draws.cos_iota[sel],
        iota=draws.iota[sel],
        optimal_snr=draws.optimal_snr[sel],
        scatter_snr=draws.scatter_snr[sel],
        log_prob_draw=log_p_draw[sel],
        log_prob_mass_1_d_mass_2_d_luminosity_distance=log_p_m1d_m2d_dL,
        num_total_injections=jnp.array([num_samples]),
        snr_threshold=jnp.array([snr_thr]),
        seed=jnp.array([seed]),
    )

    if out_path is not None:
        save_namespace_to_hdf5(cat, out_path)

    return cat


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def concatenate_namespaces(namespaces):
    """Concatenate arrays across a list of SimpleNamespace objects."""
    if not namespaces:
        raise ValueError("The input list of namespaces is empty.")

    attrs = list(vars(namespaces[0]).keys())
    result = {}

    for attr in attrs:
        values = [getattr(ns, attr) for ns in namespaces]

        arrs = []
        conversion_failed = False
        for v in values:
            try:
                a = jnp.asarray(v)
            except Exception:
                conversion_failed = True
                break
            arrs.append(a)
        if conversion_failed:
            result[attr] = values[0]
            continue

        if all(a.ndim == 0 for a in arrs):
            result[attr] = jnp.stack(arrs, axis=0)
        else:
            flat = [a.reshape(-1) if a.ndim > 0 else a.reshape(1) for a in arrs]
            try:
                result[attr] = jnp.concatenate(flat, axis=0)
            except Exception:
                result[attr] = values[0]

    return SimpleNamespace(**result)
