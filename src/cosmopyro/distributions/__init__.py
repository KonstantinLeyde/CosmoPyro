from typing import TYPE_CHECKING

from .._lazy import lazy_getattr as _lazy_getattr

if TYPE_CHECKING:
    from .source_frame_masses import construct_source_frame_mass_model

_EXPORTS = {
    "construct_source_frame_mass_model": ".source_frame_masses",
}

__all__ = [
    "construct_source_frame_mass_model",
]

__getattr__ = _lazy_getattr(globals(), _EXPORTS)
