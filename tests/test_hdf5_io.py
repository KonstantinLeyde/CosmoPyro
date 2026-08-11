import os
from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

# Import your functions here.
# Assuming your script is named `hdf5_utils.py`
from cosmopyro.data.data_utils import load_hdf5_to_namespace, save_namespace_to_hdf5


# --- Helper for comparing complex structures ---
def assert_structures_equal(obj1, obj2):
    """
    Recursively compares two objects (SimpleNamespace, dict, or array).
    Note: The loader always converts dicts to SimpleNamespace, so we
    normalize obj1 (the original) to SimpleNamespace for comparison if needed.
    """
    # Normalize dicts to SimpleNamespace for comparison symmetry
    if isinstance(obj1, dict):
        obj1 = SimpleNamespace(**obj1)
    if isinstance(obj2, dict):
        obj2 = SimpleNamespace(**obj2)

    # Compare SimpleNamespaces
    if isinstance(obj1, SimpleNamespace) and isinstance(obj2, SimpleNamespace):
        d1 = vars(obj1)
        d2 = vars(obj2)
        assert d1.keys() == d2.keys(), f"Keys mismatch: {d1.keys()} vs {d2.keys()}"
        for k in d1:
            assert_structures_equal(d1[k], d2[k])

    # Compare Arrays (NumPy or JAX)
    elif isinstance(obj1, (np.ndarray, jnp.ndarray)) or isinstance(
        obj2, (np.ndarray, jnp.ndarray)
    ):
        # Convert JAX arrays to Numpy for comparison
        a1 = np.asarray(obj1)
        a2 = np.asarray(obj2)

        if a1.ndim == 0 or a2.ndim == 0:
            assert a1 == a2
        else:
            np.testing.assert_array_equal(a1, a2)

    # Compare Scalars/Primitives
    else:
        assert obj1 == obj2, f"Value mismatch: {obj1} vs {obj2}"


# --- Test Cases ---


def test_basic_roundtrip(tmp_path):
    """Test saving and loading a flat SimpleNamespace with arrays and scalars."""
    file_path = tmp_path / "test_basic.h5"

    data = SimpleNamespace(
        array_data=np.array([1, 2, 3, 4, 5]), scalar_float=10.5, scalar_int=42
    )

    save_namespace_to_hdf5(data, file_path)
    loaded_data = load_hdf5_to_namespace(file_path)

    assert_structures_equal(data, loaded_data)


def test_nested_structure(tmp_path):
    """Test deeply nested combinations of SimpleNamespace and dicts."""
    file_path = tmp_path / "test_nested.h5"

    # Note: Your loader converts dicts back to SimpleNamespace,
    # so we expect the loaded structure to be purely SimpleNamespace.
    data = SimpleNamespace(
        level1=SimpleNamespace(val=1, level2={"matrix": np.eye(3), "deep_scalar": 99}),
        flat_val=0,
    )

    save_namespace_to_hdf5(data, file_path)
    loaded_data = load_hdf5_to_namespace(file_path)

    # Manual checks to ensure structure depth is preserved
    assert loaded_data.level1.val == 1
    assert loaded_data.level1.level2.deep_scalar == 99
    np.testing.assert_array_equal(loaded_data.level1.level2.matrix, np.eye(3))


def test_jax_array_support(tmp_path):
    """Test that JAX arrays are converted and saved correctly."""
    file_path = tmp_path / "test_jax.h5"

    jax_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    data = SimpleNamespace(jax_data=jax_arr)

    save_namespace_to_hdf5(data, file_path)
    loaded_data = load_hdf5_to_namespace(file_path)

    # Check if data matches
    np.testing.assert_array_equal(loaded_data.jax_data, np.array(jax_arr))
    # Loaded data comes back as numpy array from HDF5
    assert isinstance(loaded_data.jax_data, np.ndarray)


def test_overwrite_protection(tmp_path):
    """Ensure FileExistsError is raised when overwrite=False."""
    file_path = tmp_path / "test_overwrite.h5"
    data = SimpleNamespace(x=1)

    # Save once
    save_namespace_to_hdf5(data, file_path)

    # Try to save again without overwrite flag
    with pytest.raises(FileExistsError):
        save_namespace_to_hdf5(data, file_path, overwrite=False)


def test_overwrite_success(tmp_path):
    """Ensure file is updated when overwrite=True."""
    file_path = tmp_path / "test_overwrite_ok.h5"

    data1 = SimpleNamespace(x=1)
    save_namespace_to_hdf5(data1, file_path)

    data2 = SimpleNamespace(x=999)  # New data
    save_namespace_to_hdf5(data2, file_path, overwrite=True)

    loaded = load_hdf5_to_namespace(file_path)
    assert loaded.x == 999


def test_custom_group(tmp_path):
    """Test saving to a specific HDF5 group/path."""
    file_path = tmp_path / "test_group.h5"
    data = SimpleNamespace(y=np.zeros(5))
    group_name = "/experiment/run_1"

    save_namespace_to_hdf5(data, file_path, group=group_name)

    # Load back specifically from that group
    loaded = load_hdf5_to_namespace(file_path, group=group_name)
    np.testing.assert_array_equal(loaded.y, np.zeros(5))

    # Verify using raw h5py that the structure is correct
    with h5py.File(file_path, "r") as f:
        assert "experiment" in f
        assert "run_1" in f["experiment"]
        assert "y" in f["experiment"]["run_1"]


def test_unsupported_type(tmp_path):
    """Test that unsupported types raise a TypeError."""
    file_path = tmp_path / "test_bad_type.h5"

    # Functions are not supported by h5py/numpy serialization
    def my_func():
        pass

    data = SimpleNamespace(bad_item=my_func)

    with pytest.raises(TypeError):
        save_namespace_to_hdf5(data, file_path)

    # Ensure cleanup happened (file shouldn't exist or should be empty/deleted)
    # Your code creates the file context `with h5py.File...`, so if it crashes inside,
    # the file might remain but be empty or partial.
    # Your code explicitly has `os.remove` in the except block, let's verify that.
    assert not os.path.exists(file_path)


def test_scalar_types(tmp_path):
    """Test specific handling of 0-D arrays vs pure python scalars."""
    file_path = tmp_path / "test_scalars.h5"

    data = SimpleNamespace(
        py_float=3.14, np_float=np.float64(3.14), py_int=10, np_int=np.int32(10)
    )

    save_namespace_to_hdf5(data, file_path)
    loaded = load_hdf5_to_namespace(file_path)

    assert loaded.py_float == 3.14
    assert loaded.np_float == 3.14
    assert loaded.py_int == 10
    assert loaded.np_int == 10
