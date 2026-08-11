import inspect
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .jax_utils import compute_centers_and_delta_from_array

__all__ = [
    "KWARGS_ANALYSIS_DEFAULT",
    "apply_mask_to_dict",
    "check_cosmology_has_high_enough_redshift_coverage",
    "check_data_arrays_finite",
    "check_num_events",
    "check_priors",
    "check_source_frame_masses_within_grid_bounds",
    "clean_and_convert_to_floats",
    "collect_issues_from_priors",
    "concatenate_dicts_of_jnp_arrays",
    "construct_args_from_function_and_kwargs",
    "construct_array_from_df",
    "construct_array_from_list_df",
    "df_to_dict_of_np_array_and_labels",
    "filter_columns_with_dynamic_range",
    "flatten_dict",
    "get_binning_from_kwargs_analysis",
    "get_error_messages_luminosity_distance_within_grid",
    "get_filename_from_path",
    "is_latent_sample_site",
    "is_svi_initialization",
    "load_yaml_file",
    "run_data_checks",
    "seperate_gibbs_params",
    "xarray_to_dict",
]


def construct_array_from_df(df, keys=None, keys_exclude=None, sort_keys=True):
    """
    Constructs a NumPy array from specified columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        keys (list): List of column names to extract from the DataFrame.

    Returns:
        np.ndarray: A NumPy array containing the data from the specified columns.
    """
    # Extract specified columns and convert to NumPy array
    if keys is None:
        keys = df.columns.tolist()

    if sort_keys:
        keys_sorted = sorted(keys)
    else:
        keys_sorted = keys

    keys_filtered = []
    if keys_exclude is not None:
        for key in keys_sorted:
            if key not in keys_exclude:
                keys_filtered.append(key)
        keys_sorted = keys_filtered

    array = df[keys_sorted].to_numpy()

    return array, keys_sorted


def construct_array_from_list_df(list_df, thetas):
    """
    Constructs a stacked JAX array from a list of DataFrames and a list of parameter names.

    Parameters:
        list_df (list of pd.DataFrame): List of pandas DataFrames to extract data from.
        thetas (list of str): List of column names to extract from each DataFrame.

    Returns:
        jnp.array: Stacked JAX array with data corresponding to `thetas` from `list_df`.
    """
    # Extract relevant columns from each DataFrame
    extracted_arrays = [df[thetas].values for df in list_df]

    # Stack the arrays along a new axis (0)
    stacked_array = jnp.stack([jnp.array(arr) for arr in extracted_arrays])

    return stacked_array


def df_to_dict_of_np_array_and_labels(df, labels=None):

    if labels is None:
        labels = df.columns.tolist()
    arr = np.array([df[col] for col in labels])

    return arr, labels


def concatenate_dicts_of_jnp_arrays(list_of_dicts):

    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *list_of_dicts)


def is_latent_sample_site(site):
    """Return True for sites that HMC/NUTS samples in unconstrained space."""
    if site.get("type") != "sample":
        return False
    if site.get("is_observed", False):
        return False
    fn = site.get("fn", None)
    if fn is not None and type(fn).__name__ == "Unit":
        return False
    return True


def seperate_gibbs_params(params_list_init, not_dense_params=None, gibbs_sites=None):
    """
    Separate parameters into dense-mass groups and Gibbs blocks.

    Parameters
    ----------
    params_list_init : list of str
        All sample variable names from the model trace.
    not_dense_params : list of str, optional
        Parameters to exclude from the dense mass matrix.
    gibbs_sites : list of str, optional
        Parameters to sample via Gibbs (MH) rather than NUTS.
        If empty or None, all params are sampled jointly by NUTS.

    Returns
    -------
    list_dense_params : list
        Dense mass specification for the NUTS kernel.
    gibbs_sites_out : list of str
        Sites to be sampled via Gibbs. Empty list if no Gibbs.
    """
    not_dense_params = not_dense_params or []
    gibbs_sites = gibbs_sites or []

    # Filter out log_prob, log_det, and not_dense_params
    params_list = [
        p
        for p in params_list_init
        if not p.startswith("log_prob")
        and not p.startswith("log_det")
        and p not in not_dense_params
    ]

    # Resolve _base aliases: if a configured gibbs site isn't in the trace
    # directly but its reparametrized form '<site>_base' is, use that instead.
    def _resolve(site):
        if site in params_list_init:
            return site
        base = site + "_base"
        if base in params_list_init:
            return base
        return None

    gibbs_sites_resolved = [r for s in gibbs_sites if (r := _resolve(s)) is not None]

    # Remove gibbs_sites from dense mass
    nuts_params = [p for p in params_list if p not in gibbs_sites_resolved]
    list_dense_params = [[tuple(nuts_params)]] if nuts_params else [[]]
    gibbs_sites_out = gibbs_sites_resolved

    if gibbs_sites_out:
        print(f"Gibbs sites (MH): {gibbs_sites_out}")
        print(f"NUTS sites (dense mass): {nuts_params}")
    else:
        print(f"Dense mass params: {nuts_params}")

    return list_dense_params, gibbs_sites_out


def clean_and_convert_to_floats(nested_dict):
    """
    Recursively process a nested dictionary to:
    - Remove JAX arrays of length 0.
    - Convert numeric values to floats.
    - Leave string values unchanged.

    Args:
        nested_dict (dict): The input dictionary containing JAX arrays, lists, or nested dicts.

    Returns:
        dict: A cleaned dictionary with processed values.
    """

    def process_value(value):
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            return clean_and_convert_to_floats(value)
        elif isinstance(value, jnp.ndarray):
            # Skip arrays of length 0, convert single-element arrays to float
            return float(value.item()) if value.size == 1 else None
        elif isinstance(value, (int, float)):
            # Convert integers and floats to float
            return float(value)
        elif isinstance(value, str):
            # Leave strings unchanged
            return value
        else:
            # Skip unsupported types
            return None

    # Build a new dictionary with processed values
    return {
        key: result
        for key, value in nested_dict.items()
        if (result := process_value(value)) is not None
    }


def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def filter_columns_with_dynamic_range(df):
    """
    Filter columns of a DataFrame that have a dynamic range.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing only columns with a dynamic range.
    """
    dynamic_columns = []
    for col in df.columns:
        if df[col].max() != df[col].min():
            dynamic_columns.append(col)
    return df[dynamic_columns]


def apply_mask_to_dict(dictionary, mask):
    return {key: value[mask] for key, value in dictionary.items()}


def get_filename_from_path(path):
    """
    Extracts the filename from a given path.

    Args:
    path (str): The full path to the file.

    Returns:
    str: The filename extracted from the path.
    """
    return os.path.basename(path)


def xarray_to_dict(data):
    data_dict = {}

    # Iterate over data variables and their values
    for var_name, var_data in data.data_vars.items():
        # Convert data variable to numpy array if it's numeric
        if jnp.issubdtype(var_data.dtype, jnp.number):
            data_dict[var_name] = jnp.asarray(var_data)

    return data_dict


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def construct_args_from_function_and_kwargs(func, kwargs):
    """
    Constructs a dictionary of arguments for a given function based on provided keyword arguments.

    Parameters:
        func (callable): The target function for which to construct arguments.
        kwargs (dict): A dictionary of keyword arguments.

    Returns:
        dict: A dictionary containing only the arguments that match the function's parameters.
    """
    sig = inspect.signature(func)
    bound_args = sig.bind(**kwargs)
    bound_args.apply_defaults()
    return bound_args.args


def check_priors(priors, path="kwargs_priors"):

    errors = collect_issues_from_priors(priors, path=path)
    if errors:
        for error in errors:
            print(error)
        raise ValueError("Prior definitions contain errors. See messages above.")
    else:
        print("All prior definitions passed the checks.")


def collect_issues_from_priors(priors, path="kwargs_priors"):
    """
    Recursively check prior definitions for common consistency issues.

    Returns a list of human-readable error/warning strings.
    """
    issues = []

    def _is_number(x):
        return isinstance(x, (int, float))

    def _walk(d, path):
        if not isinstance(d, dict):
            return

        # Leaf node with a distribution
        if "dist_type" in d:
            dist = d["dist_type"]

            # ---- Uniform ----
            if dist == "Uniform":
                if "min" not in d or "max" not in d:
                    issues.append(f"[ERROR] {path}: Uniform prior missing min or max")
                else:
                    mn, mx = d["min"], d["max"]
                    if not (_is_number(mn) and _is_number(mx)):
                        issues.append(
                            f"[ERROR] {path}: Uniform min/max must be numeric"
                        )
                    elif not mn < mx:
                        issues.append(
                            f"[ERROR] {path}: Uniform prior has min >= max ({mn} >= {mx})"
                        )

            # ---- Delta ----
            elif dist == "Delta":
                if "value" not in d:
                    issues.append(f"[ERROR] {path}: Delta prior missing value")
                else:
                    val = d["value"]
                    if not _is_number(val):
                        issues.append(f"[ERROR] {path}: Delta value must be numeric")

                    # Optional consistency check if min/max exist
                    if "min" in d and val < d["min"]:
                        issues.append(
                            f"[ERROR] {path}: Delta value {val} < min {d['min']}"
                        )
                    if "max" in d and val > d["max"]:
                        issues.append(
                            f"[ERROR] {path}: Delta value {val} > max {d['max']}"
                        )

            # ---- Normal ----
            elif dist == "Normal":
                if "loc" not in d or "scale" not in d:
                    issues.append(f"[ERROR] {path}: Normal prior missing loc or scale")
                else:
                    loc, scale = d["loc"], d["scale"]
                    if not _is_number(loc):
                        issues.append(f"[ERROR] {path}: Normal loc must be numeric")
                    if not _is_number(scale) or scale <= 0:
                        issues.append(f"[ERROR] {path}: Normal scale must be positive")

            elif dist == "LogUniform":
                if "min" not in d or "max" not in d:
                    issues.append(
                        f"[ERROR] {path}: LogUniform prior missing min or max"
                    )
                else:
                    mn, mx = d["min"], d["max"]
                    if not (_is_number(mn) and _is_number(mx)):
                        issues.append(
                            f"[ERROR] {path}: LogUniform min/max must be numeric"
                        )
                    elif not (0 < mn < mx):
                        issues.append(
                            f"[ERROR] {path}: LogUniform prior must have 0 < min < max ({mn} < {mx})"
                        )

            # ---- Unknown distribution ----
            else:
                issues.append(f"[WARNING] {path}: Unknown dist_type '{dist}'")

        # Recurse deeper
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v, f"{path}.{k}")

    _walk(priors, path)
    return issues


def run_data_checks(analysis, data_kwargs):
    errors = check_cosmology_has_high_enough_redshift_coverage(analysis, data_kwargs)
    errors += check_source_frame_masses_within_grid_bounds(analysis, data_kwargs)
    errors += check_data_arrays_finite(data_kwargs)
    errors += check_num_events(data_kwargs)

    skymap = data_kwargs.get("skymap", None)
    if skymap is not None:
        import jax.numpy as jnp

        from ..data.build_skymap import validate_skymap

        nside = analysis.nside
        try:
            validate_skymap(skymap, analysis.binning, nside)
        except ValueError as e:
            errors.append(str(e))

        # Check that prob_skyposition_zhp sums to a constant across healpixels
        row_sums = jnp.sum(skymap.prob_skyposition_zhp, axis=-1)
        row_sum_dev = float(jnp.max(jnp.abs(row_sums - row_sums[0])))
        if row_sum_dev > 1e-6:
            errors.append(
                f"Sky map rows (prob_skyposition_zhp.sum(axis=-1)) are not constant. "
                f"Max deviation across redshift bins: {row_sum_dev:.2e}."
            )

    if errors:
        for error in errors:
            print(error)
        raise ValueError("Data checks failed. See messages above.")
    else:
        print("All data checks passed.")


def check_data_arrays_finite(data_kwargs):
    """Check that PE samples and injections contain no NaN or Inf values."""
    import jax.numpy as jnp

    errors = []
    required_attrs = [
        "mass_1_d",
        "mass_ratio",
        "luminosity_distance",
        "prior_masses_d_dL",
    ]
    data = data_kwargs.get("data", None)
    if data is None:
        return errors

    for label, container in [
        ("PE samples", getattr(data, "samples", None)),
        ("injections", getattr(data, "injections", None)),
    ]:
        if container is None:
            continue
        for attr in required_attrs:
            if not hasattr(container, attr):
                errors.append(f"{label}: missing required attribute '{attr}'.")
                continue
            v = getattr(container, attr)
            n_nan = int(jnp.isnan(v).sum())
            n_inf = int(jnp.isinf(v).sum())
            if n_nan > 0:
                errors.append(f"{label}.{attr} contains {n_nan} NaN values.")
            if n_inf > 0:
                errors.append(f"{label}.{attr} contains {n_inf} Inf values.")

    return errors


def _get_prior_min_max(prior, label):
    if "value" in prior or prior.get("dist_type", None) == "Delta":
        value = prior.get("value", None)
        if value is None:
            raise ValueError(f"Could not determine fixed value for {label}.")
        return value, value
    if "min" in prior and "max" in prior:
        return prior["min"], prior["max"]
    raise ValueError(f"Could not determine min/max for {label} from priors.")


def _get_hubble_prior_min_max(cosmology_priors):
    if "H0" in cosmology_priors:
        return _get_prior_min_max(cosmology_priors["H0"], "cosmology.H0")
    if "h" in cosmology_priors:
        h_min, h_max = _get_prior_min_max(cosmology_priors["h"], "cosmology.h")
        return 100.0 * h_min, 100.0 * h_max
    raise ValueError("Could not determine H0 from cosmology priors.")


def _get_cosmology_prior_corner_parameters(analysis):
    cosmology_priors = analysis.kwargs_analysis["kwargs_priors"]["cosmology"]
    H0_min, H0_max = _get_hubble_prior_min_max(cosmology_priors)

    if "Omega_m" in cosmology_priors:
        Omega_m_min, Omega_m_max = _get_prior_min_max(
            cosmology_priors["Omega_m"],
            "cosmology.Omega_m",
        )
    else:
        print("Could not determine Omega_m from priors. Using default value 0.3.")
        Omega_m_min, Omega_m_max = 0.3, 0.3

    fixed_parameters = {
        key: prior["value"]
        for key, prior in cosmology_priors.items()
        if prior.get("dist_type", None) == "Delta" and key not in ["h", "H0"]
    }

    corners = []
    for H0 in sorted({H0_min, H0_max}):
        for Omega_m in sorted({Omega_m_min, Omega_m_max}):
            parameters = dict(fixed_parameters)
            parameters.update({"H0": H0, "Omega_m": Omega_m})
            corners.append(dict(cosmology=parameters))

    return corners


def _get_grid_bounds(analysis, key):
    if hasattr(analysis, "binning") and key in analysis.binning.get("boundaries", {}):
        boundaries = analysis.binning["boundaries"][key]
        return float(boundaries[0]), float(boundaries[-1])

    if key in analysis.kwargs_analysis.get("bins", {}):
        bin_kwargs = analysis.kwargs_analysis["bins"][key]
        return float(bin_kwargs["min"]), float(bin_kwargs["max"])

    raise ValueError(f"Could not determine grid bounds for '{key}'.")


def _has_grid_bounds(analysis, key):
    if hasattr(analysis, "binning") and key in analysis.binning.get("boundaries", {}):
        return True
    return key in analysis.kwargs_analysis.get("bins", {})


def _get_grid_bound_errors(
    values, label, bounds, context, percentile_min=0.0, percentile_max=100.0
):
    lower, upper = bounds
    value_min = float(jnp.percentile(values, percentile_min))
    value_max = float(jnp.percentile(values, percentile_max))
    errors = []

    if value_min < lower:
        errors.append(
            f"{label}: minimum {value_min:.6g} is below {context} grid minimum "
            f"{lower:.6g}."
        )
    if value_max > upper:
        errors.append(
            f"{label}: maximum {value_max:.6g} exceeds {context} grid maximum "
            f"{upper:.6g}."
        )

    return errors


def check_source_frame_masses_within_grid_bounds(analysis, data_kwargs):
    """
    Check derived source-frame mass coordinates over cosmology prior corners.
    Remark: because injections cover very high masses, we check only up to the 99.9
    percentile for mass_1_s to allow for a few outliers.
    """
    errors = []
    data = data_kwargs.get("data", None)
    if data is None:
        return errors

    check_m1_q = _has_grid_bounds(analysis, "mass_1_s") and _has_grid_bounds(
        analysis,
        "mass_ratio",
    )
    check_logM_delta = _has_grid_bounds(
        analysis,
        "log_mass_total_s",
    ) and _has_grid_bounds(analysis, "minus_log_mass_ratio")

    if not check_m1_q and not check_logM_delta:
        raise ValueError(
            "Could not determine mass grid bounds. Expected either "
            "('mass_1_s', 'mass_ratio') or "
            "('log_mass_total_s', 'minus_log_mass_ratio')."
        )

    if check_m1_q:
        mass_1_s_bounds = _get_grid_bounds(analysis, "mass_1_s")
        mass_ratio_bounds = _get_grid_bounds(analysis, "mass_ratio")
    if check_logM_delta:
        log_mass_total_s_bounds = _get_grid_bounds(analysis, "log_mass_total_s")
        minus_log_mass_ratio_bounds = _get_grid_bounds(
            analysis,
            "minus_log_mass_ratio",
        )

    cosmology_corner_parameters = _get_cosmology_prior_corner_parameters(analysis)

    for data_label, samples in [
        ("PE samples", getattr(data, "samples", None)),
        ("injections", getattr(data, "injections", None)),
    ]:
        if samples is None:
            continue

        for attr in ["mass_1_d", "mass_ratio", "luminosity_distance"]:
            if not hasattr(samples, attr):
                errors.append(f"{data_label}: missing required attribute '{attr}'.")
                continue
        missing_required_attr = any(
            not hasattr(samples, attr)
            for attr in ["mass_1_d", "mass_ratio", "luminosity_distance"]
        )
        if missing_required_attr:
            continue

        if check_m1_q:
            errors += _get_grid_bound_errors(
                samples.mass_ratio,
                f"{data_label}.mass_ratio",
                mass_ratio_bounds,
                "mass_ratio",
            )

        if check_logM_delta:
            minus_log_mass_ratio = -jnp.log(samples.mass_ratio)
            errors += _get_grid_bound_errors(
                minus_log_mass_ratio,
                f"{data_label}.minus_log_mass_ratio",
                minus_log_mass_ratio_bounds,
                "minus_log_mass_ratio",
            )

        for parameters in cosmology_corner_parameters:
            cosmological_model = analysis.get_cosmological_model(parameters)
            redshift = cosmological_model.get_redshift_from_luminosity_distance(
                samples.luminosity_distance,
            )
            mass_1_s = samples.mass_1_d / (1 + redshift)
            mass_2_s = samples.mass_1_d * samples.mass_ratio / (1 + redshift)

            cosmology_context = (
                f"at H0={parameters['cosmology']['H0']:.6g}, "
                f"Omega_m={parameters['cosmology']['Omega_m']:.6g}"
            )

            if check_m1_q:
                errors += _get_grid_bound_errors(
                    mass_1_s,
                    f"{data_label}.mass_1_s",
                    mass_1_s_bounds,
                    f"mass_1_s {cosmology_context}",
                    percentile_max=99.9,  # some injections go to 1000
                )

            if check_logM_delta:
                log_mass_total_s = jnp.log(mass_1_s + mass_2_s)
                errors += _get_grid_bound_errors(
                    log_mass_total_s,
                    f"{data_label}.log_mass_total_s",
                    log_mass_total_s_bounds,
                    f"log_mass_total_s {cosmology_context}",
                    percentile_max=99.9,  # some injections go to 1000
                )

            if not jnp.all(mass_2_s <= mass_1_s):
                errors.append(
                    f"{data_label}: derived mass_2_s exceeds mass_1_s for "
                    f"{cosmology_context}; check mass_ratio values."
                )

    return errors


def check_num_events(data_kwargs):
    """Check that injections.num_events is defined, positive, and consistent."""
    errors = []
    data = data_kwargs.get("data", None)
    if data is None or not hasattr(data, "injections"):
        return errors

    inj = data.injections
    if not hasattr(inj, "num_events"):
        errors.append(
            "injections.num_events is not defined. This is required for selection effect correction."
        )
    else:
        n = inj.num_events
        if n is None or n <= 0:
            errors.append(f"injections.num_events must be positive, got {n}.")

    return errors


def get_error_messages_luminosity_distance_within_grid(
    dL_max, z_max_grid, cosmological_model, parameters
):

    # TODO this does not cover MG models, but we can add that in the future if needed
    dL_max_grid = cosmological_model.get_luminosity_distance_from_redshift(z_max_grid)
    cosmology_parameters = parameters["cosmology"]
    context = (
        f"H0={cosmology_parameters['H0']:.6g}, "
        f"Omega_m={cosmology_parameters['Omega_m']:.6g}"
    )

    errors = []
    if dL_max > dL_max_grid:
        errors += [
            f"Warning: maximum luminosity distance in the data ({dL_max:.2f} Mpc) "
            f"exceeds the maximum luminosity distance in the grid "
            f"({dL_max_grid:.2f} Mpc) for {context}. "
            f"Consider increasing z_max_grid."
        ]
    else:
        print(
            f"Passing check: maximum luminosity distance in the data "
            f"({dL_max:.2f} Mpc) is within the grid limit "
            f"({dL_max_grid:.2f} Mpc) for {context}."
        )

    return errors


def check_cosmology_has_high_enough_redshift_coverage(analysis, data_kwargs):

    z_max_grid = analysis.kwargs_analysis["cosmology_numerics"].get("z_max", None)

    if z_max_grid is None:
        raise ValueError("Could not find z_max_grid value. ")

    cosmology_corner_parameters = _get_cosmology_prior_corner_parameters(analysis)

    errors = []
    for a in ["samples", "injections"]:
        samples = getattr(data_kwargs["data"], a, None)
        if samples is None or not hasattr(samples, "luminosity_distance"):
            continue
        dL_max = samples.luminosity_distance.max()

        for parameters in cosmology_corner_parameters:
            cosmological_model = analysis.get_cosmological_model(parameters)
            errors += get_error_messages_luminosity_distance_within_grid(
                dL_max,
                z_max_grid,
                cosmological_model,
                parameters=parameters,
            )

    return errors


KWARGS_ANALYSIS_DEFAULT = dict(
    bins=dict(
        mass_1_s=dict(
            min=5.0,
            max=100.0,
            num=200,
        ),
        mass_ratio=dict(
            min=0.05,
            max=1.0,
            num=200,
        ),
        log_mass_total_s=dict(
            min=1.5,
            max=5.5,
            num=200,
        ),
        minus_log_mass_ratio=dict(
            min=0.0,
            max=3.0,
            num=200,
        ),
    ),
)


def get_binning_from_kwargs_analysis(kwargs_analysis=None, discretization_3d=None):

    if kwargs_analysis is None:
        kwargs_analysis = KWARGS_ANALYSIS_DEFAULT

    binning = dict(
        boundaries={},
    )

    if discretization_3d is not None:
        binning["boundaries"] = {}
        binning["boundaries"]["redshift"] = discretization_3d.boundaries["r"]
        binning["boundaries"]["healpix_idx"] = discretization_3d.boundaries[
            "healpix_idx"
        ]

    for key in ["mass_1_s", "mass_ratio", "log_mass_total_s", "minus_log_mass_ratio"]:
        if key in kwargs_analysis["bins"]:
            binning["boundaries"][key] = jnp.linspace(
                kwargs_analysis["bins"][key]["min"],
                kwargs_analysis["bins"][key]["max"],
                kwargs_analysis["bins"][key]["num"] + 1,
            )

    binning["centers"], binning["deltas"] = {}, {}
    for key in binning["boundaries"].keys():
        binning["centers"][key], binning["deltas"][key] = (
            compute_centers_and_delta_from_array(binning["boundaries"][key])
        )

    return binning


def is_svi_initialization(kwargs_sampler):
    return (
        kwargs_sampler.get("num_svi_samples", 0) != 0
        and kwargs_sampler.get("num_svi_steps", 0) != 0
    )
