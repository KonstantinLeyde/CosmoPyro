import os
from types import SimpleNamespace

import astropy.units as u
import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..utils.jax_utils import (
    group_sizes_are_uniform,
    make_segment_ids_from_group_sizes,
)
from ..utils.transformations import get_component_masses_from_chirp_mass_and_mass_ratio

__all__ = [
    "EXCEPTED_EVENTS",
    "downselect_posterior_samples_and_priors",
    "get_distance_priors_from_event_dict",
    "get_prior_o4a",
    "load_all_events_from_event_dict",
    "load_events_from_dict",
    "overwrite_if_event_in_exceptions",
    "post_process_catalog",
]


def _load_hdf5_node(node):
    """
    Recursively load HDF5 groups/datasets into a dict or DataFrame.
    Adjust this based on your actual data structure (e.g. if leaf nodes are arrays).
    """
    if isinstance(node, h5py.Group):
        d = {}
        for key, item in node.items():
            d[key] = _load_hdf5_node(item)
        # If this group looks like a table (has specific columns), convert to DataFrame
        if "mass_1" in d:
            return pd.DataFrame(d)
        return d
    elif isinstance(node, h5py.Dataset):
        return node[()]
    return None


def load_events_from_dict(event_dict, valid_types=None, debug=False):
    """
    Load datasets specified in event_dict.

    Parameters
    ----------
    event_dict : dict
        The dictionary containing event metadata (filename, type, samples_field).
    valid_types : list
        List of strings for allowed event types (e.g. ['BBH', 'BNS']).
    debug : bool
        If True, stop after loading one successful event.

    Returns
    -------
    dict
        Mapping event_name -> loaded object
    list
        List of event names successfully loaded
    """
    results = {}
    errors = []

    for event_name, metadata in event_dict.items():
        print(f"Processing event: {event_name}")

        # 1. Filter by Event Type
        e_type = metadata.get("event_type", "Unknown")
        if valid_types is not None and e_type not in valid_types:
            continue

        path = metadata.get("posterior_file_path")
        target = metadata.get("samples_field")

        if not os.path.exists(path):
            errors.append(f"{event_name}: File not found at {path}")
            continue

        # 3. Load HDF5
        try:
            with h5py.File(path, "r") as hdf:
                if target not in hdf:
                    # Try to find a partial match or print keys for debugging
                    print(
                        f"Key '{target}' not found in {event_name}. Available: {list(hdf.keys())}"
                    )
                    raise KeyError(f"Target '{target}' not found in {filename}")

                res = _load_hdf5_node(hdf[target])
                results[event_name] = downselect_posterior_samples_and_priors(res)

        except Exception as e:
            errors.append(f"{event_name}: {e}")
            continue

        if debug:
            print(f"[DEBUG] Loaded {event_name} from {path}")
            break

    if errors:
        print(f"Encountered {len(errors)} errors:")
        for e in errors[:5]:  # print first 5 errors
            print(" -", e)
        raise Exception("Errors encountered during loading. See above for details.")

    return results


def _choose_indices(available, n, rng):
    return rng.choice(available, size=n, replace=False)


def _num_to_use(option, num_posterior_samples, available):
    if option == "fixed":
        if num_posterior_samples is None:
            raise ValueError(
                "`num_posterior_samples` must be provided when option='fixed'."
            )
        if available < num_posterior_samples:
            raise ValueError(
                f"Requested {num_posterior_samples} samples but only {available} are available."
            )
        return num_posterior_samples

    if option == "num_or_available":
        return (
            available
            if num_posterior_samples is None
            else min(num_posterior_samples, available)
        )

    raise ValueError(f"Invalid option: {option}")


def post_process_catalog(
    catalog,
    *,
    distance_priors_dict,
    num_posterior_samples=None,
    option="num_or_available",
    seed=98823452,
    ignore_names=None,
):
    if ignore_names is None:
        ignore_names = []

    valid_names = [n for n in catalog if n not in ignore_names]
    if not valid_names:
        print("No valid events found to post-process.")
        return SimpleNamespace()

    params_input = ["chirp_mass", "mass_ratio", "luminosity_distance", "ra", "dec"]
    params = ["chirp_mass_d", "mass_ratio", "luminosity_distance", "ra", "dec"]
    rename_map = {"chirp_mass": "chirp_mass_d"}

    samples_df = {
        n: pd.DataFrame(catalog[n]["posterior_samples"][params_input]).rename(
            columns=rename_map
        )
        for n in valid_names
    }

    rng = np.random.default_rng(seed)
    idxs, priors = {}, []
    n_per_event = []

    for name in valid_names:
        df = samples_df[name]
        available = len(df)
        n_use = _num_to_use(option, num_posterior_samples, available)
        n_per_event.append(n_use)
        idxs[name] = _choose_indices(available, n_use, rng)

        d_lum = df["luminosity_distance"].to_numpy()[idxs[name]]
        prior_type = distance_priors_dict[name]

        if prior_type == "uniform_masses_d_luminosity_distance_squared":
            w = d_lum**2
        elif prior_type == "uniform_masses_d_luminosity_distance_uniform_source_frame":
            w = get_prior_o4a().prob(d_lum)
        else:
            raise NotImplementedError(f"Prior type {prior_type} not implemented.")

        priors.append(w / w.sum())

    stack_func = (
        np.hstack if option == "num_or_available" else lambda xs: np.stack(xs, axis=0)
    )

    d = SimpleNamespace()
    d.prior_masses_d_dL = stack_func(priors)

    for k in params:
        vals = [samples_df[name][k].to_numpy()[idxs[name]] for name in valid_names]
        setattr(d, k, stack_func(vals))

    d.num_posterior_samples_per_event = jnp.asarray(n_per_event, dtype=jnp.int64)
    d.posterior_sample_groups_are_uniform = group_sizes_are_uniform(
        d.num_posterior_samples_per_event
    )
    if not d.posterior_sample_groups_are_uniform:
        d.posterior_sample_segment_ids = make_segment_ids_from_group_sizes(
            d.num_posterior_samples_per_event
        )
    return d


def downselect_posterior_samples_and_priors(d):
    dd = {k: v for k, v in d.items() if k in ["posterior_samples", "priors"]}
    return dd


def load_all_events_from_event_dict(
    event_dict, input_type="BBH", num_posterior_samples=10_000, debug=True
):
    """
    Main function to load events based on the input dictionary and type filter.

    Parameters
    ----------
    event_dict : dict
        The JSON dictionary provided.
    input_type : str
        String specifying types to load, e.g. "BBH", "BNS", "BBH+NSBH".
    debug : bool
        Debug flag.
    """

    # Parse the input type string (e.g. "BBH+NSBH" -> ['BBH', 'NSBH'])
    valid_types = [t.strip() for t in input_type.split("+")]

    print(f"Loading events of type: {valid_types}...")

    # Load data
    catalog_dict = load_events_from_dict(
        event_dict, valid_types=valid_types, debug=debug
    )

    print(f"Loaded {len(catalog_dict)} events.")

    distance_priors_dict = get_distance_priors_from_event_dict(event_dict, catalog_dict)

    # Post process (simplenamespace)
    samples_sn = post_process_catalog(
        catalog_dict,
        distance_priors_dict=distance_priors_dict,
        num_posterior_samples=num_posterior_samples,
        option="num_or_available",
    )

    samples_sn.mass_1_d, _ = get_component_masses_from_chirp_mass_and_mass_ratio(
        samples_sn.chirp_mass_d, samples_sn.mass_ratio
    )
    samples_sn.names = [n.encode("ascii", "ignore") for n in catalog_dict.keys()]

    return samples_sn


def overwrite_if_event_in_exceptions(event_name):

    if event_name in EXCEPTED_EVENTS:
        return EXCEPTED_EVENTS[event_name]
    return None


def get_distance_priors_from_event_dict(event_dict, catalog_df):

    prior_dict_events = {}

    for k in catalog_df.keys():
        print("Adding distance prior for event:", k)

        overwritten = overwrite_if_event_in_exceptions(k)
        if overwritten is not None:
            prior_dict_events[k] = overwritten
            print(prior_dict_events)
            continue

        if "priors" in catalog_df[k].keys():
            priors = catalog_df[k]["priors"]["analytic"]
        else:
            raise KeyError(f"Event {k} missing 'priors' specification.")

        if "luminosity_distance" not in priors:
            raise KeyError(
                f"Event {k} missing 'luminosity_distance' in 'analytic' prior specification."
            )

        prior_string = str(priors["luminosity_distance"])

        if "PowerLaw(alpha=2" in prior_string:
            prior_dict_events[k] = "uniform_masses_d_luminosity_distance_squared"
        elif "bilby.gw.prior.UniformSourceFrame(minimum=" in prior_string:
            prior_dict_events[k] = (
                "uniform_masses_d_luminosity_distance_uniform_source_frame"
            )
        else:
            raise NotImplementedError(
                f"Distance prior {prior_string} for event {k} not implemented."
            )

    return prior_dict_events


def get_prior_o4a():
    import bilby
    from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_astropy

    # from https://git.ligo.org/benoit.revenu/gwcosmo-fork/-/blob/lscsoft_priors/gwcosmo/likelihood/posterior_samples.py#L70
    # self.cosmo = astropy.cosmology.Planck15 # CAREFUL: Planck15 in astropy is NOT the LVK reference which is PLanck 2015 TT+lowP+lensing+ext cosmology
    # we use the values for H0 and Om0 reported in https://dcc.ligo.org/DocDB/0167/T2000185/005/LVC_symbol_convention.pdf and https://zenodo.org/records/6513631
    cosmo = FlatLambdaCDM_astropy(
        H0=67.90 * (u.km / u.s / u.Mpc),
        Om0=0.3065,
        Tcmb0=2.7255 * u.K,
        Neff=3.046,
        m_nu=[0.0, 0.0, 0.06] * u.eV,
        Ob0=0.0486,
    )

    return bilby.gw.prior.UniformSourceFrame(
        minimum=0.1,
        maximum=50000.0,
        cosmology=cosmo,
        name="luminosity_distance",
        latex_label="$d_L$",
        unit="Mpc",
        boundary=None,
    )


EXCEPTED_EVENTS = {
    "GW170608_020116": "uniform_masses_d_luminosity_distance_squared",
    "GW170817_124104": "uniform_masses_d_luminosity_distance_squared",
    "GW190425_081805": "uniform_masses_d_luminosity_distance_squared",
    "GW190707_093326": "uniform_masses_d_luminosity_distance_squared",
    "GW190720_000836": "uniform_masses_d_luminosity_distance_squared",
    "GW190725_174728": "uniform_masses_d_luminosity_distance_squared",
    "GW190728_064510": "uniform_masses_d_luminosity_distance_squared",
    "GW190814_211039": "uniform_masses_d_luminosity_distance_squared",
    "GW190917_114630": "uniform_masses_d_luminosity_distance_squared",
    "GW190924_021846": "uniform_masses_d_luminosity_distance_squared",
}
