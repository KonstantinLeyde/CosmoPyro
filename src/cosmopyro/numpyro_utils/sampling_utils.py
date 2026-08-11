from contextlib import ExitStack

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam

__all__ = [
    "get_Delta_samples",
    "get_Dirichlet_samples",
    "get_LogUniform_samples",
    "get_Normal_samples",
    "get_Uniform_samples",
    "get_prior_dict_samples",
    "get_prior_draw",
    "plates_for_shape",
]


def _as_tuple_shape(shape):
    if shape is None:
        return None
    return (shape,) if isinstance(shape, int) else tuple(shape)


def plates_for_shape(base_name, shape):
    shape = _as_tuple_shape(shape)
    stack = ExitStack()
    if shape is None:
        return stack
    for i, size in enumerate(shape[::-1]):
        stack.enter_context(numpyro.plate(f"{base_name}_plate_{i}", size, dim=-1 - i))
    return stack


# --- Sampling Functions with Internalized Reparam ---


def get_Uniform_samples(name, prior_dict_info, reparam=True):
    low, high = prior_dict_info["min"], prior_dict_info["max"]
    shape = prior_dict_info.get("shape", None)

    if reparam:
        # Explicit pattern required by newer NumPyro:
        # Uniform(low, high) = AffineTransform(low, high - low) applied to Uniform(0, 1)
        # NUTS then samples in the unconstrained base space.
        base = dist.Uniform(0.0, 1.0)
        d = dist.TransformedDistribution(base, AffineTransform(low, high - low))
        with numpyro.handlers.reparam(config={name: TransformReparam()}):
            with plates_for_shape(name, shape):
                return numpyro.sample(name, d)
    else:
        with plates_for_shape(name, shape):
            return numpyro.sample(name, dist.Uniform(low, high))


def get_LogUniform_samples(name, prior_dict_info, reparam=True):
    dist_min, dist_max = prior_dict_info["min"], prior_dict_info["max"]
    shape = prior_dict_info.get("shape", None)

    stack = ExitStack()
    if reparam:
        stack.enter_context(numpyro.handlers.reparam(config={name: TransformReparam()}))

    with stack, plates_for_shape(name, shape):
        return numpyro.sample(name, dist.LogUniform(dist_min, dist_max))


def get_Normal_samples(name, prior_dict_info):
    mu, sigma = prior_dict_info["loc"], prior_dict_info["scale"]
    shape = prior_dict_info.get("shape", None)
    with plates_for_shape(name, shape):
        eps = numpyro.sample(f"{name}_white", dist.Normal(0, 1))
        samples = mu + sigma * eps
        return numpyro.deterministic(name, samples)


def get_Delta_samples(name, prior_dict_info, use_value_for_Delta=True):
    if use_value_for_Delta:
        return prior_dict_info["value"]
    shape = prior_dict_info.get("shape", None)
    with plates_for_shape(name, shape):
        return numpyro.sample(name, dist.Delta(v=prior_dict_info["value"]))


def get_Dirichlet_samples(name, prior_dict_info):
    concentration = prior_dict_info["concentration"]
    shape = prior_dict_info.get("shape", None)
    with plates_for_shape(name, shape):
        return numpyro.sample(name, dist.Dirichlet(concentration))


# --- Modern Dispatcher ---


def get_prior_dict_samples(param, prior_dict_info, use_value_for_Delta=True):
    dist_type = prior_dict_info.get("dist_type", "Uniform")

    # Structural Pattern Matching (Python 3.10+)
    match dist_type:
        case "Uniform":
            return get_Uniform_samples(param, prior_dict_info)
        case "LogUniform":
            return get_LogUniform_samples(param, prior_dict_info)
        case "Normal":
            return get_Normal_samples(param, prior_dict_info)
        case "Delta":
            return get_Delta_samples(param, prior_dict_info, use_value_for_Delta)
        case "Dirichlet":
            return get_Dirichlet_samples(param, prior_dict_info)
        case _:
            raise NotImplementedError(f"Distribution '{dist_type}' is not supported.")


def get_prior_draw(prior, use_value_for_Delta=True):
    params = {}
    for key, sub_prior in prior.items():
        if sub_prior is None:
            continue
        params[key] = {
            p_name: get_prior_dict_samples(p_name, p_info, use_value_for_Delta)
            for p_name, p_info in sub_prior.items()
        }
    return params
