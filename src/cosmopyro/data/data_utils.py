import os
from collections.abc import Mapping
from types import SimpleNamespace

import h5py
import numpy as np

__all__ = [
    "load_hdf5_to_namespace",
    "save_namespace_to_hdf5",
]


def _save_recursive(data, hf_group, compression="gzip", compression_opts=4):
    """
    Recursively saves SimpleNamespace/dict content to an HDF5 group.
    """
    # Determine how to iterate over the data (SimpleNamespace or dict)
    if hasattr(data, "__dict__"):
        items = data.__dict__.items()
    elif isinstance(data, Mapping):
        items = data.items()
    else:
        raise TypeError(
            f"Unsupported container type: {type(data)}. Must be SimpleNamespace or dict."
        )

    for k, v in items:
        # 1. Handle Nested Structures (SimpleNamespace or dict)
        if isinstance(v, (SimpleNamespace, Mapping)):
            new_group = hf_group.require_group(k)
            _save_recursive(v, new_group, compression, compression_opts)
            continue

        # Prepare value for saving: Convert lists/tuples/JAX arrays to NumPy arrays
        try:
            v = np.asarray(v)
        except Exception as e:
            raise TypeError(
                f"Unsupported type for key '{k}': {type(v)}. Could not convert to array."
            ) from e

        # 2. Handle Scalars (0-D arrays)
        if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
            # Scalars do not support chunking/filters, so we only pass data
            hf_group.create_dataset(k, data=v)
            continue

        # 3. Handle NumPy/JAX Arrays (N-D arrays)
        if isinstance(v, np.ndarray):
            # Store with compression, chunking, checksum, and shuffling (only for N-D data)
            hf_group.create_dataset(
                k,
                data=v,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,  # Improves compression for float data
                fletcher32=True,  # Checksum for data integrity
            )
            continue

        # 4. Fallback (Should have been caught by np.asarray, but for safety)
        raise TypeError(f"Unsupported final type for key '{k}': {type(v)}")


def save_namespace_to_hdf5(ns, path, group="/", overwrite=False, **kwargs):
    """
    Save a namespace or dict of numpy arrays/scalars (including nested structures)
    to an HDF5 file.

    Args:
        ns (SimpleNamespace or dict): The data structure to save.
        path (str): The filepath for the HDF5 file.
        group (str): The root group name within the HDF5 file (default is '/').
        overwrite (bool): If False (default), raise FileExistsError when path
                  already exists. Pass True to replace the file.
        **kwargs: Optional arguments passed to the recursive saver (e.g., compression,
                  compression_opts).
    """
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"The file '{path}' already exists. To prevent overwriting, please choose a different path or delete the existing file."
            )

    print(f"Saving structure to {path} at group '{group}'...")
    try:
        with h5py.File(path, "w") as f:
            root_group = f.require_group(group)
            _save_recursive(ns, root_group, **kwargs)
        print("Save successful.")
    except Exception as e:
        print(f"Error during save: {e}")
        # Clean up corrupted file if an error occurred during writing
        if os.path.exists(path):
            os.remove(path)
        raise


def _load_recursive(hf_item):
    """
    Recursively loads HDF5 group content into a SimpleNamespace structure.
    """
    result_ns = SimpleNamespace()

    for k, v in hf_item.items():
        if isinstance(v, h5py.Group):
            # Nested Group -> Nested SimpleNamespace
            setattr(result_ns, k, _load_recursive(v))
        elif isinstance(v, h5py.Dataset):
            # Dataset -> Load data
            # v[()] loads the entire dataset content into memory
            setattr(result_ns, k, v[()])

    return result_ns


def load_hdf5_to_namespace(path, group="/"):
    """
    Load data from an HDF5 file back into a SimpleNamespace structure,
    including nested namespaces.

    Args:
        path (str): The filepath for the HDF5 file.
        group (str): The root group name to load from (default is '/').

    Returns:
        SimpleNamespace: The reconstructed data structure.
    """
    print(f"Loading structure from {path} at group '{group}'...")
    with h5py.File(path, "r") as f:
        if group not in f:
            raise KeyError(f"Group '{group}' not found in HDF5 file.")

        root_data = f[group]
        loaded_ns = _load_recursive(root_data)
        print("Load successful.")
        return loaded_ns
