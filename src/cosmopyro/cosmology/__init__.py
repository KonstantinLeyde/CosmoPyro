from typing import TYPE_CHECKING

from .._lazy import lazy_getattr as _lazy_getattr

if TYPE_CHECKING:
    from .cosmology import get_cosmological_model
    from .flat_lambdacdm import C_KM_PER_SEC, FlatLambdaCDM

_EXPORTS = {
    "C_KM_PER_SEC": ".flat_lambdacdm",
    "FlatLambdaCDM": ".flat_lambdacdm",
    "get_cosmological_model": ".cosmology",
}

__all__ = [
    "C_KM_PER_SEC",
    "FlatLambdaCDM",
    "get_cosmological_model",
]

__getattr__ = _lazy_getattr(globals(), _EXPORTS)
