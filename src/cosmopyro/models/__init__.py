from typing import TYPE_CHECKING

from .._lazy import lazy_getattr as _lazy_getattr

if TYPE_CHECKING:
    from .models import (
        calculate_safe_log_prob_batched,
        get_log_prob,
        model_evaluate_p_theta,
    )

_EXPORTS = {
    "calculate_safe_log_prob_batched": ".models",
    "get_log_prob": ".models",
    "model_evaluate_p_theta": ".models",
}

__all__ = [
    "calculate_safe_log_prob_batched",
    "get_log_prob",
    "model_evaluate_p_theta",
]

__getattr__ = _lazy_getattr(globals(), _EXPORTS)
