import pickle

import numpy as np

__all__ = [
    "convert_arviz_to_numpy_dict",
    "divide_scalars_and_arrays_from_dict",
    "filter_columns_with_dynamic_range",
    "flatten_dict",
    "flatten_dict_along_chain_dim",
    "get_ith_entry_of_array",
    "last_state_read",
    "last_state_write",
    "squeeze_dict",
]


def convert_arviz_to_numpy_dict(inf_data, skip_vars_end_with="_base"):
    """
    Convert ArviZ data variables to a dictionary of NumPy arrays.

    Parameters:
    - inf_data (arviz.InferenceData): ArviZ InferenceData object containing posterior samples.

    Returns:
    - numpy_dict (dict): Dictionary containing selected data variables as NumPy arrays.
    """

    # Extract data variables from inf_data
    data_vars = inf_data.posterior.data_vars

    # Initialize an empty dictionary to store NumPy arrays
    numpy_dict = {}

    # Iterate over data variables and convert them to NumPy arrays
    for var_name, var_data in data_vars.items():
        # Skip variables ending with 'base'
        if var_name.endswith(skip_vars_end_with):
            continue

        # Check if the data type of the variable is integer or float
        if np.issubdtype(var_data.dtype, np.integer) or np.issubdtype(
            var_data.dtype, np.floating
        ):
            # Extract the data as a NumPy array
            var_array = np.asarray(var_data)
            # Add the NumPy array to the dictionary with the variable name as key
            numpy_dict[var_name] = var_array

    # add also predictions

    # Extract data variables from inf_data
    try:
        data_vars = inf_data.posterior_predictive.data_vars
    except:
        data_vars = {}

    # Iterate over data variables and convert them to NumPy arrays
    for var_name, var_data in data_vars.items():
        # Check if the data type of the variable is integer or float
        if np.issubdtype(var_data.dtype, np.integer) or np.issubdtype(
            var_data.dtype, np.floating
        ):
            # Extract the data as a NumPy array
            var_array = np.asarray(var_data)
            # Add the NumPy array to the dictionary with the variable name as key
            numpy_dict[var_name] = var_array

    return numpy_dict


def squeeze_dict(input_dict):
    return {k: v.squeeze() for k, v in input_dict.items()}


def flatten_dict_along_chain_dim(input_dict):
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in input_dict.items()}


def divide_scalars_and_arrays_from_dict(input_dict):
    return {k: v for k, v in input_dict.items() if len(v.shape) == 1}, {
        k: v for k, v in input_dict.items() if len(v.shape) != 1
    }


def get_ith_entry_of_array(dictionary, i):
    result_dict = {}
    for key, value in dictionary.items():
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f"The value for key '{key}' is not an array")
        if i < 0 or i >= len(value):
            raise IndexError(f"Index {i} out of range for key '{key}'")
        result_dict[key] = value[i]
    return result_dict


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


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary.

    Parameters:
    d (dict): The dictionary to flatten.
    parent_key (str): The base key string for recursion (used internally).
    sep (str): The separator between keys.

    Returns:
    dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def last_state_write(last_state, file_out):
    with open(file_out, "wb") as f:
        pickle.dump(last_state, f)


def last_state_read(file_in):
    with open(file_in, "rb") as f:
        last_state = pickle.load(f)
    return last_state
