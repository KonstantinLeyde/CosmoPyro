from typing import TYPE_CHECKING

from .._lazy import lazy_getattr as _lazy_getattr

if TYPE_CHECKING:
    from .build_skymap import build_skymap, validate_skymap
    from .data_utils import load_hdf5_to_namespace, save_namespace_to_hdf5
    from .generate_catalogs import generate_catalog_from_galaxy_catalog
    from .generate_simple_pe import compute_posterior_samples

_EXPORTS = {
    "build_skymap": ".build_skymap",
    "compute_posterior_samples": ".generate_simple_pe",
    "generate_catalog_from_galaxy_catalog": ".generate_catalogs",
    "load_hdf5_to_namespace": ".data_utils",
    "save_namespace_to_hdf5": ".data_utils",
    "validate_skymap": ".build_skymap",
}

__all__ = [
    "build_skymap",
    "compute_posterior_samples",
    "generate_catalog_from_galaxy_catalog",
    "load_hdf5_to_namespace",
    "save_namespace_to_hdf5",
    "validate_skymap",
]

__getattr__ = _lazy_getattr(globals(), _EXPORTS)
