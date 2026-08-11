from typing import TYPE_CHECKING

from .._lazy import lazy_getattr as _lazy_getattr

if TYPE_CHECKING:
    from .sampling_utils import get_prior_draw

_EXPORTS = {
    "get_prior_draw": ".sampling_utils",
}

__all__ = [
    "get_prior_draw",
]

__getattr__ = _lazy_getattr(globals(), _EXPORTS)
