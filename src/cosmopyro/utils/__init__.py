from typing import TYPE_CHECKING

from .._lazy import lazy_getattr as _lazy_getattr

if TYPE_CHECKING:
    from .analyses import Analysis, Result
    from .analysis_utils import find_differences, find_results
    from .helper_functions import last_state_read
    from .jax_utils import compute_centers_and_delta_from_array
    from .plotting import (
        make_mass_plot,
        make_mass_plot_s_delta,
        plot_posterior_comparison,
    )
    from .runtime_utilities import get_config, get_time_stamp
    from .spherical_coordinates import HealPixDiscretization3D
    from .utils import get_binning_from_kwargs_analysis, is_latent_sample_site

_EXPORTS = {
    "Analysis": ".analyses",
    "HealPixDiscretization3D": ".spherical_coordinates",
    "Result": ".analyses",
    "compute_centers_and_delta_from_array": ".jax_utils",
    "find_differences": ".analysis_utils",
    "find_results": ".analysis_utils",
    "get_binning_from_kwargs_analysis": ".utils",
    "get_config": ".runtime_utilities",
    "get_time_stamp": ".runtime_utilities",
    "is_latent_sample_site": ".utils",
    "last_state_read": ".helper_functions",
    "make_mass_plot": ".plotting",
    "make_mass_plot_s_delta": ".plotting",
    "plot_posterior_comparison": ".plotting",
}

__all__ = [
    "Analysis",
    "HealPixDiscretization3D",
    "Result",
    "compute_centers_and_delta_from_array",
    "find_differences",
    "find_results",
    "get_binning_from_kwargs_analysis",
    "get_config",
    "get_time_stamp",
    "is_latent_sample_site",
    "last_state_read",
    "make_mass_plot",
    "make_mass_plot_s_delta",
    "plot_posterior_comparison",
]

__getattr__ = _lazy_getattr(globals(), _EXPORTS)
