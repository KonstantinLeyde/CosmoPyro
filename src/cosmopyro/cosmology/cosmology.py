from .flat_lambdacdm import FlatLambdaCDM
from .modified_gw_distance_ratio import (
    ModifiedGWDistanceFlatLambdaCDM,
    ModifiedGWLuminosityDistanceCosmology,
)

__all__ = [
    "FlatLambdaCDM",
    "get_cosmological_model",
]


def get_cosmological_model(name, cosmology_numerics, parameters):
    params_cosmology = parameters["cosmology"]

    if name == "FlatLambdaCDM":
        cosmological_model = FlatLambdaCDM(
            params=params_cosmology,
            numerics=cosmology_numerics,
        )
    elif name in [
        "FlatLambdaCDM_GW_distance_polynomial",
        "FlatLambdaCDM_GW_distance_cosine",
        "FlatLambdaCDM_GW_distance_gp_integrated",
        "FlatLambdaCDM_GW_distance_cM",
    ]:
        cosmological_model = ModifiedGWDistanceFlatLambdaCDM(
            params=params_cosmology,
            name_modified_ratio=name,
            params_modified_gravity=parameters.get("modified_ratio"),
            numerics=cosmology_numerics,
        )
    elif name in ["FlatLambdaCDM_gp_integrated_dLGW"]:
        cosmological_model = ModifiedGWLuminosityDistanceCosmology(
            params=params_cosmology,
            name=name,
            params_modified_gravity=parameters.get("modified_luminosity_distance_GW"),
            numerics=cosmology_numerics,
        )
    else:
        raise ValueError(f"Unknown cosmology model name: {name}")

    return cosmological_model
