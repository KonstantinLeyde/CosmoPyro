import arviz as az
import blackjax
import jax
import numpy as np
from numpyro.infer.util import initialize_model

__all__ = [
    "convert_to_arviz",
    "from_numpyro",
    "process_in_chunks",
    "run_mclmc",
    "run_mclmc_model",
    "setup_model",
]


def from_numpyro(model, rng_key, model_args):
    # 1. We explicitely capture 'postprocess_fn' instead of hiding it in *_
    init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        rng_key,
        model,
        model_args=model_args,
        dynamic_args=True,
    )

    # 2. Create the log density function
    # potential_fn returns Energy (Negative Log Likelihood), so we negate it.
    potential_fn = potential_fn_gen(*model_args)
    logdensity_fn = lambda position: -potential_fn(position)

    # 3. Create a ready-to-use transform function
    # The raw postprocess_fn expects args first, then the position.
    # We wrap it here so you only need to pass the position later.
    def transform_fn(position):
        return postprocess_fn(*model_args)(position)

    initial_position = init_params.z

    return logdensity_fn, initial_position, transform_fn


def run_mclmc(
    logdensity_fn,
    num_steps,
    initial_position,
    key,
    transform,
    desired_energy_variance=5e-4,
):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    # build the kernel
    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    # find values for L and step_size
    (blackjax_state_after_tuning, blackjax_mclmc_sampler_params, _) = (
        blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=False,
            desired_energy_var=desired_energy_variance,
        )
    )

    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
    )

    # run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )

    return samples, blackjax_state_after_tuning, blackjax_mclmc_sampler_params, run_key


def setup_model(model, model_args, seed):

    rng_key = jax.random.key(seed)

    # Now you get the transform_fn back
    logp_sv, x_init, transform_fn = from_numpyro(model, rng_key, model_args)

    return logp_sv, x_init, transform_fn


def run_mclmc_model(model, model_args, num_steps, seed, desired_energy_variance):

    logp_sv, x_init, transform_fn = setup_model(model, model_args, seed)

    rng_key = jax.random.key(seed + 1)
    samples_raw, blackjax_state_after_tuning, blackjax_mclmc_sampler_params, run_key = (
        run_mclmc(
            logdensity_fn=logp_sv,
            num_steps=num_steps,
            initial_position=x_init,
            key=rng_key,
            transform=lambda state, _: state.position,
            desired_energy_variance=desired_energy_variance,
        )
    )

    # jax.lax.map applies the function to the leading dimension (0) sequentially
    # samples = jax.lax.map(transform_fn, samples_raw)
    # the alternative below produces OOM errors
    # samples = jax.vmap(transform_fn)(samples_raw)
    samples = process_in_chunks(samples_raw, transform_fn)

    return samples, blackjax_state_after_tuning, blackjax_mclmc_sampler_params, run_key


def process_in_chunks(samples_raw, transform_fn):
    samples = jax.lax.map(transform_fn, samples_raw)
    return samples


def convert_to_arviz(samples):
    # 1. Convert JAX arrays to Numpy (ArviZ prefers Numpy)
    #    and ensure (chains, draws, ...) shape.
    posterior = {}

    for key, val in samples.items():
        val_np = np.array(val)

        # CASE A: It's a single chain (draws, ...)
        # We need to add the chain dimension -> (1, draws, ...)
        if val_np.ndim == 1:
            # Turn (2000,) into (1, 2000)
            posterior[key] = val_np[None, :]
        else:
            # Turn (2000, 5) into (1, 2000, 5)
            posterior[key] = val_np[None, ...]

    return az.from_dict(posterior=posterior)
