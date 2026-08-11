import copy
import os
import warnings
from pprint import pformat
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro

numpyro.enable_x64()

import arviz as az
import numpy as np
import optax
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, HMCGibbs, Predictive, log_likelihood

from .. import __version__
from ..cosmology.cosmology import get_cosmological_model
from ..data.data_utils import (
    load_hdf5_to_namespace,
    save_namespace_to_hdf5,
)
from ..data.load_catalogs import load_posterior_samples_and_injections_from_file
from ..distributions.redshift import get_redshift_model
from ..distributions.source_frame_masses import (
    construct_source_frame_mass_model,
)
from ..numpyro_utils import blackjax_utils
from ..utils.jax_utils import (
    sum_log_probs,
)
from ..utils.utils import (
    check_priors,
    construct_args_from_function_and_kwargs,
    get_binning_from_kwargs_analysis,
    is_latent_sample_site,
    is_svi_initialization,
    run_data_checks,
)
from .class_utils import update_dict
from .helper_functions import (
    convert_arviz_to_numpy_dict,
    flatten_dict_along_chain_dim,
    last_state_read,
    last_state_write,
)
from .jupyter_formatting import print_summary_in_notebook
from .plotting import plot_trace_scalar_variables
from .spherical_coordinates import (
    HealPixDiscretization3D,
)
from .svi_utils import (
    build_guide,
    inverse_mass_matrix_from_guide_samples,
    inverse_mass_matrix_from_hessian,
    subset_data,
)
from .utils import get_filename_from_path, seperate_gibbs_params, xarray_to_dict

__all__ = [
    "Analysis",
    "Result",
    "SkeletonAnalysis",
    "get_redshift_discretization_kwargs",
    "load_cosmopyro_results_no_tensors",
    "pick_out_add_healpix_discretization_3d_kwargs",
    "pick_out_apparent_magnitude_discretization_kwargs",
    "remove_tensors_from_arviz",
]


class SkeletonAnalysis:
    def __init__(
        self,
        model,
        settings_path=None,
        job_id=None,
        start_from_subresult_i=0,
        can_overwrite_kwargs=True,
        root_folder=None,
    ):
        if settings_path is None:
            raise ValueError("Please provide a path to the settings file.")

        self._kwargs_analysis = None

        self.model = model
        self.settings_path = settings_path
        self.start_from_subresult_i = start_from_subresult_i
        self.can_overwrite_kwargs = can_overwrite_kwargs
        self.root_folder = root_folder

        self.load_kwargs_analysis_from_path(settings_path)
        self.job_id = self.get_or_read_job_id(job_id)

        self.setup_results_folder()

        if "job_id" not in self.kwargs_analysis.keys():
            # add job_id to the kwargs_analysis
            self.update_kwargs_analysis(["job_id"], self.job_id)

        self.check_version()

        self.use_svi_initialization = is_svi_initialization(
            self.kwargs_analysis["kwargs_sampler"]
        )

        if self.root_folder is not None:
            self.change_root_folder()

    def __str__(self):
        return pformat(self.kwargs_analysis, width=60, compact=True)

    @property
    def kwargs_analysis(self):
        return self._kwargs_analysis

    @property
    def nb_subresults(self):
        if "num_posterior_samples" in self.kwargs_analysis["kwargs_sampler"]:
            return self.kwargs_analysis["kwargs_sampler"][
                "num_posterior_samples"
            ] // self.kwargs_analysis["kwargs_sampler"].get(
                "num_posterior_samples_per_batch", 1000
            )
        else:
            return 0

    def print_summary(self):
        print_summary_in_notebook(self.kwargs_analysis)

    def _repr_html_(self):
        return self.print_summary()

    def get_or_read_job_id(self, job_id):
        if job_id is None:
            try:
                j_id = self.kwargs_analysis["job_id"]
            except:
                raise ValueError("Job ID not provided. ")
        else:
            j_id = job_id

        return j_id

    def load_yaml_file(self, path):
        with open(path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)

    def load_kwargs_analysis_from_path(self, settings_path):
        if self._kwargs_analysis is not None:
            raise RuntimeError
        self._kwargs_analysis = self.load_yaml_file(settings_path)

    def change_root_folder(self):
        """
        Changes paths in the catalog metadata to local ones.

        Args:
        catalog_metadata (dict): The metadata containing paths to be changed.

        Returns:
        dict: The modified catalog metadata with local paths.
        """

        local_data_path = f"{self.root_folder}/data/"
        local_results_path = f"{self.root_folder}/results/"

        catalog_metadata = self.kwargs_analysis["catalog_metadata"]

        # Change results path to local
        catalog_metadata["results_path"] = local_results_path
        catalog_metadata["data_folder"] = local_data_path

        catalog_full_filename = catalog_metadata["catalog_full_filename"]

        # Change catalog full filename to local
        new_catalog_full_filename = os.path.join(
            local_data_path, get_filename_from_path(catalog_full_filename)
        )
        catalog_metadata["catalog_full_filename"] = new_catalog_full_filename
        self.kwargs_analysis["catalog_settings_filename"] = os.path.join(
            local_data_path,
            get_filename_from_path(self.kwargs_analysis["catalog_settings_filename"]),
        )

        self.kwargs_analysis["results_path"] = local_results_path

        print("Changed catalog paths to local. ")

    def update_kwargs_analysis(self, keys, value):

        if not self.can_overwrite_kwargs:
            raise ValueError("Cannot update read-only kwargs. ")

        update_dict(keys, value, self._kwargs_analysis)
        self.save_kwargs_analysis(self.get_results_folder())

    def check_version(self):
        """Stamp the cosmopyro version into the settings, warning on mismatch.

        Settings written by an earlier run carry the version that produced them,
        so re-running or loading results with a different install is flagged
        rather than silently reinterpreted.
        """
        version_kwargs = self.kwargs_analysis.get("cosmopyro_version", None)

        if version_kwargs is None:
            if self.can_overwrite_kwargs:
                self.update_kwargs_analysis(["cosmopyro_version"], __version__)
            else:
                warnings.warn(
                    f"These settings carry no cosmopyro version, so they predate "
                    f"version tracking; cosmopyro {__version__} is installed. ",
                    stacklevel=2,
                )
        elif version_kwargs != __version__:
            warnings.warn(
                f"Version mismatch: these settings were written with cosmopyro "
                f"{version_kwargs}, but cosmopyro {__version__} is installed. "
                f"Results may not be reproducible. ",
                stacklevel=2,
            )

    def setup_results_folder(self):
        results_path = self.get_results_folder()

        for f in ["preliminary", "plots", "samples"]:
            rp = results_path + f"/{f}"
            if rp and not os.path.exists(rp):
                os.makedirs(rp)

        if results_path:
            self.save_kwargs_analysis(results_path)

    def save_kwargs_analysis(self, results_path):
        with open(os.path.join(results_path, "kwargs_analysis.yaml"), "w") as yaml_file:
            yaml.safe_dump(self.kwargs_analysis, yaml_file)

    def get_run_kwargs(self):
        run_kwargs = self.kwargs_analysis.get("run_kwargs", None)

        if run_kwargs is None:
            raise "No settings specified"

        return run_kwargs

    def load_data(self):
        pass

    def run_svi_initialization(self, guide=None, save_samples=True):

        if guide is None:
            guide = build_guide(self.model, self.kwargs_analysis["kwargs_sampler"])

        max_iterations = self.kwargs_analysis["kwargs_sampler"]["num_svi_steps"]

        # Optionally use fewer events for SVI (faster, coarser initialization)
        num_events_svi = self.kwargs_analysis["kwargs_sampler"].get(
            "num_events_svi", None
        )
        svi_data_kwargs = self.data_kwargs
        if num_events_svi is not None:
            sub_data = subset_data(self.data_kwargs["data"], num_events_svi)
            svi_data_kwargs = {**self.data_kwargs, "data": sub_data}

        scheduler = optax.exponential_decay(
            init_value=0.01, decay_rate=0.99, transition_steps=max_iterations // 80
        )
        optim = optax.adabelief(learning_rate=scheduler)
        loss = numpyro.infer.Trace_ELBO()
        svi = numpyro.infer.SVI(self.model, guide, optim, loss)

        rng_key = jax.random.key(6)
        svi_result = svi.run(
            rng_key,
            max_iterations,
            analysis=self,
            **svi_data_kwargs,
            progress_bar=True,
            stable_update=True,
        )
        self.plot_svi_losses(svi_result.losses, max_iterations)

        num_svi_samples = self.kwargs_analysis["kwargs_sampler"]["num_svi_samples"]
        guide_predictive = Predictive(
            guide, params=svi_result.params, num_samples=num_svi_samples
        )
        guide_samples = guide_predictive(
            jax.random.key(1), analysis=self, **self.data_kwargs
        )

        # Run guide samples through the model to get transformed parameters
        # (e.g. h from h_base via TransformReparam). Use mask=False to skip
        # likelihood evaluation — we only need the parameter transforms.
        masked_model = numpyro.handlers.mask(self.model, mask=False)
        model_predictive = Predictive(
            masked_model, posterior_samples=guide_samples, batch_ndims=1
        )
        model_samples = model_predictive(
            jax.random.key(2),
            analysis=self,
            **self.data_kwargs,
        )
        # Merge: guide has _base sites, model has transformed sites + deterministics.
        # Keep parameters (ndim <= 2) to include both scalars and the GP field,
        # but drop large grid deterministics (ndim > 2).
        posterior_samples = {**guide_samples, **model_samples}
        posterior_samples = {k: v for k, v in posterior_samples.items() if v.ndim >= 1}

        # insert chain dimension
        posterior_samples = {k: v[None, ...] for k, v in posterior_samples.items()}

        # find posterior samples with largest posterior probability
        def get_log_prob(sample_dict):
            log_joint = log_likelihood(
                self.model,
                analysis=self,
                **self.data_kwargs,
                posterior_samples=sample_dict,
                batch_ndims=2,
            )
            total_log_prob = sum_log_probs(log_joint, batch_ndims=2)

            return total_log_prob

        total_log_prob = get_log_prob(posterior_samples)

        best_idx = jnp.unravel_index(jnp.argmax(total_log_prob), total_log_prob.shape)

        # Extract that specific sample from the dictionary
        best_sample = {k: v[best_idx] for k, v in posterior_samples.items()}

        if save_samples:
            posterior_samples_sn = SimpleNamespace(**posterior_samples)
            posterior_samples_sn.log_likelihood = total_log_prob
            save_namespace_to_hdf5(
                posterior_samples_sn,
                self.get_guide_samples_path(),
                group="/guide_samples",
            )

        return posterior_samples, best_sample

    def plot_svi_losses(self, losses, max_iterations):

        fig, ax = plt.subplots(figsize=(15, 3.5))
        ax.plot(losses)
        ax.set_yscale("asinh")

        axins = ax.inset_axes([0.3, 0.5, 0.64, 0.45])
        N_end = max_iterations // 3
        x_plot = np.linspace(max_iterations - N_end, max_iterations, N_end)
        axins.plot(x_plot, losses[max_iterations - N_end :])
        ax.indicate_inset_zoom(axins, edgecolor="k")

        fig.savefig(self.get_results_folder() + "./plots/svi_result.png")

    def configure_mcmc(
        self,
        init_strategy=None,
        not_dense_params=None,
        guide_samples=None,
        mode_for_hessian=None,
        adapt_mass_matrix=None,
    ):
        if init_strategy is None:
            init_strategy = numpyro.infer.initialization.init_to_sample

        if not_dense_params is None:
            not_dense_params = [
                "gaussian_F_whitened_spatial_white"
            ]  # TODO, maybe all vectorized inputs?

        exec_trace = trace(seed(self.model, jax.random.key(0))).get_trace(
            analysis=self,
            **self.data_kwargs,
        )
        sample_vars = [
            key for key, value in exec_trace.items() if is_latent_sample_site(value)
        ]

        gibbs_sites_config = self.kwargs_analysis["kwargs_sampler"].get(
            "gibbs_sites", []
        )

        list_dense_params, gibbs_sites = seperate_gibbs_params(
            sample_vars,
            not_dense_params=not_dense_params,
            gibbs_sites=gibbs_sites_config,
        )

        # Optionally seed NUTS' inverse mass matrix from a Laplace approximation
        # at the SVI mode (best) or the SVI guide marginal variances (fallback).
        # The Laplace H⁻¹ is generally far more accurate for tightly-constrained
        # parameters, where the SVI guide tends to over- or under-estimate width.
        inverse_mass_matrix = None
        dense_groups = list_dense_params[0]
        diag_sites = [
            s for s in not_dense_params if s in sample_vars and s not in gibbs_sites
        ]

        if mode_for_hessian is not None:
            print(
                "[configure_mcmc] Computing Hessian-based inverse mass matrix at "
                "the supplied mode (Laplace approximation)..."
            )
            inverse_mass_matrix = inverse_mass_matrix_from_hessian(
                self.model,
                model_args=(),
                model_kwargs=dict(analysis=self, **self.data_kwargs),
                mode=mode_for_hessian,
                dense_groups=dense_groups,
                diag_sites=diag_sites,
            )
            print(
                f"[configure_mcmc] Hessian-based inverse mass matrix built "
                f"({len(inverse_mass_matrix)} groups)."
            )
        elif guide_samples is not None:
            site_sizes = {
                k: int(jnp.size(v["value"]))
                for k, v in exec_trace.items()
                if is_latent_sample_site(v)
            }
            inverse_mass_matrix = inverse_mass_matrix_from_guide_samples(
                guide_samples,
                dense_groups=dense_groups,
                diag_sites=diag_sites,
                site_sizes=site_sizes,
            )
            print(
                f"[configure_mcmc] Inverse mass matrix from SVI guide marginals "
                f"({len(inverse_mass_matrix)} groups)."
            )

        nuts_kwargs = dict(
            init_strategy=init_strategy,
            max_tree_depth=self.kwargs_analysis["kwargs_sampler"]["max_tree_depth"],
            dense_mass=list_dense_params[0],
            target_accept_prob=self.kwargs_analysis["kwargs_sampler"][
                "target_accept_prob"
            ],
            forward_mode_differentiation=self.kwargs_analysis["kwargs_sampler"].get(
                "forward_mode_differentiation", False
            ),
        )
        if inverse_mass_matrix is not None:
            nuts_kwargs["inverse_mass_matrix"] = inverse_mass_matrix
            # When we hand NUTS a Laplace-quality mass matrix, default to
            # freezing it. Adaptation from a 500-step warmup typically degrades
            # a good initial estimate. Override with adapt_mass_matrix=True if
            # you want NUTS to refine.
            if adapt_mass_matrix is None and mode_for_hessian is not None:
                adapt_mass_matrix = True
        if adapt_mass_matrix is not None:
            nuts_kwargs["adapt_mass_matrix"] = adapt_mass_matrix
            print(f"[configure_mcmc] adapt_mass_matrix={adapt_mass_matrix}")
        nuts_kernel = NUTS(self.model, **nuts_kwargs)

        if not gibbs_sites:
            return nuts_kernel

        # Gibbs mode: wrap NUTS with HMCGibbs using MH for gibbs_sites.
        mh_step_size = self.kwargs_analysis["kwargs_sampler"].get(
            "gibbs_mh_step_size", 0.01
        )
        num_mh_steps = self.kwargs_analysis["kwargs_sampler"].get(
            "gibbs_num_mh_steps", 10
        )

        # Build bounds dict from priors: {site: (low, high)}.
        # For _base sites (reparametrized via TransformReparam), the base distribution
        # determines the bounds — e.g. Uniform priors use base=Uniform(0,1) so
        # _base lives in [0, 1], not the original [low, high].
        priors = self.kwargs_analysis.get("kwargs_priors", {})
        site_bounds = {}
        for site in gibbs_sites:
            # Strip _base suffix for prior lookup
            prior_name = site.removesuffix("_base")
            is_base = site.endswith("_base")
            # Walk the (possibly nested) prior dict to find the site
            site_prior = None
            for group in priors.values():
                if isinstance(group, dict) and prior_name in group:
                    site_prior = group[prior_name]
                    break
            if site_prior is not None:
                dist_type = site_prior.get("dist_type")
                if dist_type in ("Uniform", "LogUniform"):
                    if is_base:
                        # TransformReparam: base dist is Uniform(0, 1)
                        site_bounds[site] = (0.0, 1.0)
                    else:
                        site_bounds[site] = (
                            float(site_prior["min"]),
                            float(site_prior["max"]),
                        )
                elif dist_type == "Delta":
                    raise ValueError(
                        f"Gibbs site '{site}' has a Delta prior — it should not be sampled."
                    )
                elif dist_type == "Normal":
                    site_bounds[site] = (-jnp.inf, jnp.inf)
                else:
                    raise NotImplementedError(
                        f"Gibbs site '{site}' has prior dist_type='{dist_type}'. "
                        f"Only 'Uniform' and 'Normal' are supported for Gibbs MH sampling."
                    )
            else:
                site_bounds[site] = (-jnp.inf, jnp.inf)

        # Capture model args/kwargs for JIT-friendly log_density evaluation
        model_kwargs = dict(analysis=self, **self.data_kwargs)

        def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
            """Multiple random-walk MH steps using numpyro.infer.util.log_density (JIT-friendly).

            Uses jax.lax.fori_loop so the body is compiled once regardless of
            num_mh_steps, avoiding O(num_mh_steps) XLA compilation time.
            Proposals are clipped to the prior support to avoid out-of-bounds samples.
            """
            from numpyro.infer.util import log_density

            sites_keys = list(gibbs_sites.keys())  # fixed at trace time

            def one_mh_step(_, state):
                rng_key, current_sites = state

                new_values = {}
                for site in sites_keys:
                    rng_key, subkey = jax.random.split(rng_key)
                    proposal = current_sites[site] + mh_step_size * jax.random.normal(
                        subkey, shape=jnp.shape(current_sites[site])
                    )
                    low, high = site_bounds[site]
                    new_values[site] = jnp.clip(proposal, low, high)

                all_current = {**hmc_sites, **current_sites}
                all_proposed = {**hmc_sites, **new_values}

                log_p_current, _ = log_density(
                    self.model, (), model_kwargs, all_current
                )
                log_p_proposed, _ = log_density(
                    self.model, (), model_kwargs, all_proposed
                )

                rng_key, accept_key = jax.random.split(rng_key)
                log_alpha = log_p_proposed - log_p_current
                accept = jnp.log(jax.random.uniform(accept_key)) < log_alpha

                new_current_sites = {
                    site: jnp.where(accept, new_values[site], current_sites[site])
                    for site in sites_keys
                }
                return rng_key, new_current_sites

            _, current_sites = jax.lax.fori_loop(
                0, num_mh_steps, one_mh_step, (rng_key, dict(gibbs_sites))
            )
            return current_sites

        kernel = HMCGibbs(
            inner_kernel=nuts_kernel,
            gibbs_fn=gibbs_fn,
            gibbs_sites=gibbs_sites,
        )
        print(
            f"Using HMCGibbs with Gibbs sites: {gibbs_sites}, MH step size: {mh_step_size:.4f}, num MH steps: {num_mh_steps}, bounds: {site_bounds}"
        )

        return kernel

    def _get_last_state_path(self, path_result=None):
        """Return path to mcmc_last_state.pkl, optionally from a different run."""
        if path_result is not None:
            return os.path.join(path_result, "preliminary", "mcmc_last_state.pkl")
        return os.path.join(
            self.get_results_folder(), "preliminary", "mcmc_last_state.pkl"
        )

    def _try_load_previous_state(self, path_result=None):
        """Try to load a previous MCMC state (mass matrix + step size).

        Looks in path_result first, then falls back to the current run's
        preliminary folder. Returns the HMCState or None.
        """
        # Try explicit path first
        if path_result is not None:
            state_path = self._get_last_state_path(path_result)
            if os.path.exists(state_path):
                print(f"Loading previous MCMC state from: {state_path}")
                return last_state_read(state_path)
            else:
                print(f"Warning: path_result given but no state found at {state_path}")

        # Fall back to own preliminary folder
        state_path = self._get_last_state_path()
        if os.path.exists(state_path):
            print(f"Loading previous MCMC state from: {state_path}")
            return last_state_read(state_path)

        return None

    def run_mcmc(self, kernel):
        num_samples = self.kwargs_analysis["kwargs_sampler"].get(
            "num_posterior_samples", 0
        )
        num_warmup_cfg = self.kwargs_analysis["kwargs_sampler"].get("num_warmup", None)
        num_posterior_samples_per_batch = self.kwargs_analysis["kwargs_sampler"].get(
            "num_posterior_samples_per_batch", 500
        )
        num_batches = num_samples // num_posterior_samples_per_batch

        # If num_warmup is not set (None or 0), try to reuse mass matrix
        # from a previous run, skipping warmup entirely.
        reuse_state = None
        if not num_warmup_cfg:
            path_result = self.kwargs_analysis["kwargs_sampler"].get(
                "start_nuts_from_previous_run_path", None
            )
            reuse_state = self._try_load_previous_state(path_result)
            if reuse_state is not None:
                num_warmup = 0
                print("Reusing mass matrix and step size from previous run (warmup=0).")
                adapt = reuse_state.adapt_state
                print(f"  step_size: {jax.tree.leaves(adapt.step_size)}")
            else:
                raise ValueError(
                    "num_warmup is not set and no previous MCMC state found. "
                    "Either set num_warmup in kwargs_sampler or provide path_result "
                    "pointing to a run with preliminary/mcmc_last_state.pkl."
                )
        else:
            num_warmup = num_warmup_cfg

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_posterior_samples_per_batch,
            progress_bar=True,
            num_chains=self.kwargs_analysis["kwargs_sampler"]["num_chains"],
            chain_method="vectorized",
        )

        # Inject the previous state so NUTS uses the loaded mass matrix + step size
        if reuse_state is not None:
            reuse_state = jax.device_put(reuse_state)
            mcmc._last_state = reuse_state
            mcmc._warmup_state = reuse_state
            mcmc.post_warmup_state = reuse_state

        for i in range(self.start_from_subresult_i, self.nb_subresults):
            print(f"Computing posterior samples for batch {i + 1} / {num_batches}.")

            if reuse_state is not None and i == self.start_from_subresult_i:
                # First batch with reused state: use the state's rng_key and
                # pass init_params so NUTS doesn't try to re-initialize the model.
                random_key = reuse_state.rng_key
                init_params = reuse_state.z
            else:
                random_key = self.get_random_key(i, mcmc)
                init_params = None

            print("Starting running MCMC")
            mcmc.run(
                random_key,
                analysis=self,
                **self.data_kwargs,
                init_params=init_params,
            )

            print("Ran chain.")
            self.save_mcmc_results(mcmc, i)

        return mcmc.get_samples()

    def run_mclmc(self, model_args=None, seed=0):

        if model_args is None:
            if self.data_kwargs is None:
                raise ValueError("Data kwargs not set. ")
            model_args = construct_args_from_function_and_kwargs(
                self.model,
                {**self.data_kwargs, "analysis": self},  # TODO maybe better fix?
            )

        num_steps = self.kwargs_analysis["kwargs_sampler"]["num_posterior_samples"]
        desired_energy_variance = float(
            self.kwargs_analysis["kwargs_sampler"].get("desired_energy_variance", 1e-9)
        )

        samples, *_ = blackjax_utils.run_mclmc_model(
            self.model, model_args, num_steps, seed, desired_energy_variance
        )

        self.save_mclmc(samples)

        return samples

    def save_mclmc(self, samples):
        idata = blackjax_utils.convert_to_arviz(samples)

        file_path = self.get_sub_results_file_name(0)
        az.to_netcdf(idata, file_path)

    def set_data_kwargs(self, **kwargs):
        self.data_kwargs = dict(**kwargs)

    def run_data_checks(self):
        pass

    def get_random_key(self, batch_idx, mcmc):
        if batch_idx == 0:
            return jax.random.key(19251605)
        elif self.start_from_subresult_i != 0:
            print("Starting from a pre-loaded run.")
            last_state = last_state_read(self._get_last_state_path())
            mcmc._last_state = jax.device_put(last_state)
            mcmc._warmup_state = jax.device_put(last_state)
            mcmc.post_warmup_state = mcmc.last_state
            return mcmc.post_warmup_state.rng_key[0, 0]
        else:
            return mcmc.post_warmup_state.rng_key

    def save_mcmc_results(self, mcmc, batch_idx):
        last_state = jax.device_get(mcmc.last_state)
        last_state_write(last_state, self._get_last_state_path())

        mcmc._states = jax.device_get(mcmc._states)
        mcmc._states_flat = jax.device_get(mcmc._states_flat)
        mcmc.post_warmup_state = mcmc.last_state

        idata = az.from_numpyro(mcmc)
        if hasattr(idata, "sample_stats") and hasattr(idata.sample_stats, "diverging"):
            print(f"#divergences: {idata.sample_stats.diverging.values.sum()}")
        else:
            print("Divergence stats not available (e.g. HMCGibbs kernel).")

        file_path = self.get_sub_results_file_name(batch_idx)
        az.to_netcdf(idata, file_path)

    def get_results_folder(self):
        return self.kwargs_analysis["results_path"] + f"/id_{self.job_id}/"

    def get_results_samples_folder(self):
        return self.get_results_folder() + "samples/"

    def get_guide_samples_path(self):
        return self.get_results_samples_folder() + "./guide_samples.hdf5"

    def get_results_file_name(self):
        return self.get_results_samples_folder() + "result.av"

    def get_sub_results_file_name(self, i):
        return self.get_results_samples_folder() + f"result_{i}.av"

    def get_plots_folder(self):
        return self.get_results_folder() + "plots/"


class Analysis(SkeletonAnalysis):
    def __init__(
        self,
        model,
        settings_path=None,
        job_id=None,
        start_from_subresult_i=0,
        catalog_settings_filename=None,
        can_overwrite_kwargs=True,
        root_folder=None,
    ):

        super().__init__(
            model=model,
            settings_path=settings_path,
            job_id=job_id,
            start_from_subresult_i=start_from_subresult_i,
            can_overwrite_kwargs=can_overwrite_kwargs,
            root_folder=root_folder,
        )

        if self.kwargs_analysis.get("catalog_metadata", None) is not None:
            self.using_catalog = True
        else:
            self.using_catalog = False

        self._discretization_3d = None

        if self.using_catalog:
            self.setup_catalog_metadata(catalog_settings_filename)
            self.add_discretization_3d()

        self.initialize_analysis()

    def setup_catalog_metadata(self, catalog_settings_filename):

        if catalog_settings_filename is None:
            # use uniform catalog, if no settings file is provided
            self.kwargs_analysis["catalog_metadata"]["catalog_full_filename"] = (
                "uniform"
            )
            self.kwargs_analysis["catalog_settings_filename"] = "uniform"

        has_catalog_settings = (
            "catalog_settings_filename" in self.kwargs_analysis.keys()
        )
        has_catalog_metadata = "catalog_metadata" in self.kwargs_analysis.keys()

        if not has_catalog_settings and not has_catalog_metadata:
            if catalog_settings_filename is None:
                print("Catalog settings filename not provided. ")
            else:
                self.update_kwargs_analysis(
                    ["catalog_settings_filename"], catalog_settings_filename
                )
                self.update_kwargs_analysis(
                    ["catalog_metadata"], self.load_yaml_file(catalog_settings_filename)
                )

        elif has_catalog_settings and not has_catalog_metadata:
            raise "This should not have happened (1). "
        elif not has_catalog_settings and has_catalog_metadata:
            raise "This should not have happened (2). "
        else:
            if catalog_settings_filename is not None:
                print(
                    f"Found settings: {self.kwargs_analysis['catalog_settings_filename']}"
                )
            else:
                pass

    def initialize_analysis(self):
        self.setup_default_priors_from_kwargs()
        check_priors(self.prior)

    def setup_default_priors_from_kwargs(self):
        self.prior = self.kwargs_analysis["kwargs_priors"]

    def prepare_catalog_data(self):
        pass

    def prepare_catalog_truth(self):
        pass

    def get_true_catalog_settings_filename(self):
        pass

    def get_distribution_name(self, key):
        return self.kwargs_analysis["distribution_names"][key]

    @property
    def nside(self):
        hpd = (self.kwargs_analysis.get("catalog_metadata") or {}).get(
            "healpix_discretization_3d"
        ) or {}
        return hpd.get("nside", 1)

    def _load_skymap(self, path_skymap):
        if path_skymap is not None:
            skymap = load_hdf5_to_namespace(path_skymap)
            npix = skymap.prob_skyposition_zhp.shape[1]
            nside = int(round((npix / 12) ** 0.5))
            if (
                self.kwargs_analysis["catalog_metadata"]["healpix_discretization_3d"]
                is None
            ):
                self.kwargs_analysis["catalog_metadata"][
                    "healpix_discretization_3d"
                ] = {}
            self.kwargs_analysis["catalog_metadata"]["healpix_discretization_3d"][
                "nside"
            ] = nside
            self.save_kwargs_analysis(self.get_results_folder())
            print(
                f"Skymap loaded: inferred nside={nside} from prob_skyposition_zhp shape {skymap.prob_skyposition_zhp.shape}."
            )
            # Rebuild discretization_3d (binning updates automatically via setter)
            self.discretization_3d = None
            self.add_discretization_3d()
        else:
            skymap = None
        return skymap

    def load_data(self):
        run_kwargs = self.get_run_kwargs()

        skymap = self._load_skymap(run_kwargs.get("path_skymap"))

        data = load_posterior_samples_and_injections_from_file(
            filepath=run_kwargs.get("path_posterior_samples"),
            filepath_injections=run_kwargs.get("path_injections"),
            num_events=run_kwargs.get("num_events"),
            num_posterior_samples=run_kwargs.get("num_posterior_samples"),
            num_injections=run_kwargs.get("num_injections"),
            nside=self.nside,
            seed=2,
        )

        return data, skymap

    def run_data_checks(self):
        run_data_checks(self, self.data_kwargs)

    @property
    def discretization_3d(self):
        return self._discretization_3d

    @discretization_3d.setter
    def discretization_3d(self, value):
        self._discretization_3d = value
        if value is not None:
            self.binning = get_binning_from_kwargs_analysis(self.kwargs_analysis, value)

    def get_discretization_3d(self):
        if self._discretization_3d is None and self.using_catalog:
            kwargs = get_redshift_discretization_kwargs(self.kwargs_analysis)
            self.discretization_3d = HealPixDiscretization3D(nside=self.nside, **kwargs)
        return self._discretization_3d

    def add_discretization_3d(self):
        self.get_discretization_3d()
        self.check_settings()

    def check_settings(self):
        r_max_cosmo_model = self.kwargs_analysis["cosmology_numerics"]["z_max"]
        r_max_catalog = self.discretization_3d.r_max

        if r_max_catalog > r_max_cosmo_model:
            raise ValueError(
                f"Catalog r_max = {r_max_catalog} > cosmology model r_max = {r_max_cosmo_model}. Please increase the cosmology model r_max. "
            )

    def get_extreme_cosmological_models(self):

        Hubble_constant_fixed = (
            self.kwargs_analysis["kwargs_priors"]["cosmology"]["H0"]["dist_type"]
            == "Delta"
        )
        cosmo_prior_dict = self.kwargs_analysis["kwargs_priors"]["cosmology"]

        if Hubble_constant_fixed:
            H0_val = cosmo_prior_dict["H0"]["value"]
            H0_min, H0_max = H0_val, H0_val
        else:
            H0_min = cosmo_prior_dict["H0"]["min"]
            H0_max = cosmo_prior_dict["H0"]["max"]

        fixed_cosmological_parameters = {
            k: v["value"]
            for k, v in cosmo_prior_dict.items()
            if "Delta" == v.get("dist_type", None)
        }

        if "Omega_m" not in fixed_cosmological_parameters:
            raise ValueError("Omega_m not found in fixed cosmological parameters. ")

        cosmological_parameters = copy.copy(fixed_cosmological_parameters)

        cosmological_parameters.update({"H0": H0_min})
        cosmological_low = self.get_cosmological_model(
            dict(cosmology=cosmological_parameters)
        )

        cosmological_parameters.update({"H0": H0_max})
        cosmological_high = self.get_cosmological_model(
            dict(cosmology=cosmological_parameters)
        )

        return cosmological_low, cosmological_high

    def get_cosmological_model(self, parameters):

        return get_cosmological_model(
            self.kwargs_analysis["cosmology_model_name"],
            self.kwargs_analysis["cosmology_numerics"],
            parameters,
        )

    def get_source_frame_mass_model(self, parameters):

        interpolation = self.kwargs_analysis["bins"].get(
            "mass_grid_interpolation", "smooth_log"
        )
        return construct_source_frame_mass_model(
            self, params=parameters, interpolation=interpolation
        )

    def get_redshift_model(self, name, cosmological_model, parameters):
        interpolation = self.kwargs_analysis["bins"].get(
            "redshift_grid_interpolation", "smooth_log"
        )
        return get_redshift_model(
            name, cosmological_model, parameters, interpolation=interpolation
        )

    def get_batch_size(self, name):
        return self.kwargs_analysis.get("likelihood_evaluation", {}).get(
            f"{name}_batch_size", 50_000_000
        )


def remove_tensors_from_arviz(ds):
    return ds[[var for var in ds.data_vars if len(ds[var].dims) == 2]]


def get_redshift_discretization_kwargs(metadata):
    binning_metadata = metadata.get("bins")
    if binning_metadata is not None:
        return dict(
            r_min=binning_metadata["redshift"]["min"],
            r_max=binning_metadata["redshift"]["max"],
            n_r=int(binning_metadata["redshift"]["num"]),
        )
    print(
        "No binning metadata found. Using default values for redshift discretization."
    )
    return dict(r_min=0.01, r_max=1.5, n_r=10)


def pick_out_add_healpix_discretization_3d_kwargs(metadata, nside=1):

    return dict(**get_redshift_discretization_kwargs(metadata), nside=int(nside))


def pick_out_apparent_magnitude_discretization_kwargs(catalog_metadata):

    try:
        d = catalog_metadata["magnitude_discretization"]
    except:
        d = catalog_metadata["magnitude_coordinates"]
    d["n_m"] = int(d["n_m"])

    return d


class Result(Analysis):
    def __init__(self, model, settings_path=None, root_folder=None):
        super().__init__(
            model, settings_path, can_overwrite_kwargs=False, root_folder=root_folder
        )
        self.inf_data = None

    def load_inf_data(
        self, paths=None, idxs=None, skip_tensors=False, skip_irrelevant_data=True
    ):

        if idxs is None:
            idxs = list(range(self.nb_subresults))

        if paths is None:
            paths = [self.get_sub_results_file_name(idx) for idx in idxs]

        self.inf_datas = []
        for i, path in enumerate(paths):
            print(
                f"Loading {i + 1} / {len(paths)} result (#total {self.nb_subresults}). "
            )

            try:
                res = az.from_netcdf(path)
            except:
                print(f"Could not load result from {path}. Skipping. ")
                break

            if skip_tensors:
                print("Omitting tensors in posterior. ")
                res.posterior = remove_tensors_from_arviz(res.posterior)

                try:
                    del res.posterior_predictive
                except:
                    print("Posterior predictive not available to omit. ")

            if skip_irrelevant_data:
                try:
                    del res.observed_data
                    del res.log_likelihood
                except:
                    pass

            self.inf_datas.append(res)

        self.inf_data = az.concat(self.inf_datas, dim="draw")

    def trace_plot_scalar_variables(self, save=True):
        fig = plot_trace_scalar_variables(self.inf_data)
        if save and fig is not None:
            fig.savefig(
                self.get_plots_folder() + "scalar_trace.png", bbox_inches="tight"
            )
        return fig

    def make_correlation_plot(self, save=True):
        az.plot_autocorr(self.inf_data)
        if save:
            plt.savefig(
                self.get_plots_folder() + "autocorrelation.png", bbox_inches="tight"
            )

    def make_diagnostic_plots(self, save=True):
        self.trace_plot_scalar_variables(save=save)
        self.make_correlation_plot(save=save)

    def load_guide_samples(self):
        guide_samples_sn = load_hdf5_to_namespace(
            self.get_guide_samples_path(), group="/guide_samples"
        )
        return vars(guide_samples_sn)

    def set_guide_samples(self, guide_samples=None):

        if guide_samples is None:
            guide_samples = self.load_guide_samples()

        # only load 1D samples (e.g. parameters), not the GP field or large grid deterministics
        guide_samples = {k: v for k, v in guide_samples.items() if v.ndim == 2}
        guide_samples = flatten_dict_along_chain_dim(guide_samples)
        guide_samples = {
            k: jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            for k, v in guide_samples.items()
        }
        self.guide_hypersamples = guide_samples

    def load_hypersamples(self, skip_diverging_samples):
        self.hypersamples = convert_arviz_to_numpy_dict(self.inf_data)

        # get rid of the chain dimension
        has_diverging = hasattr(self.inf_data, "sample_stats") and hasattr(
            self.inf_data.sample_stats, "diverging"
        )

        if skip_diverging_samples and has_diverging:
            diverging = self.inf_data.sample_stats.diverging.values

            # if using Gibbs sampler
            if len(diverging.shape) == 3:
                diverging = np.max(diverging, axis=(-1,))
            elif len(diverging.shape) == 2:
                pass
            else:
                raise NotImplementedError

            self.hypersamples = {k: v[~diverging] for k, v in self.hypersamples.items()}

        else:
            if skip_diverging_samples and not has_diverging:
                print(
                    "Warning: divergence stats not available (e.g. HMCGibbs), skipping divergence filtering."
                )
            self.hypersamples = flatten_dict_along_chain_dim(self.hypersamples)
        self.hypersamples = {k: v for k, v in self.hypersamples.items() if v.ndim == 1}
        self.hypersamples_df = pd.DataFrame(self.hypersamples)

    def prepare_posterior_samples(self):

        if self.inf_data is None:
            self.load_inf_data()

        posterior_samples = xarray_to_dict(self.inf_data.posterior)

        return posterior_samples

    def compute_main_results(
        self,
        grid_ref=None,
        posterior_samples=None,
        num_prior_samples=None,
        batch_size=None,
    ):

        if grid_ref is None:
            grid_ref = dict()

        if "mass_1_s" not in grid_ref.keys():
            grid_ref["mass_1_s"] = jnp.linspace(2 - 1e-3, 120 + 1e-3, 500)
        if "mass_ratio" not in grid_ref.keys():
            grid_ref["mass_ratio"] = jnp.linspace(0.02, 1, 90)

        if "redshift" not in grid_ref.keys():
            grid_ref["redshift"] = jnp.linspace(
                self.kwargs_analysis["cosmology_numerics"]["z_min"],
                self.kwargs_analysis["cosmology_numerics"]["z_max"],
                200,
            )
        if "healpix_idx" not in grid_ref.keys():
            grid_ref["healpix_idx"] = self.discretization_3d.centers["healpix_idx"]

        if posterior_samples is None and num_prior_samples is None:
            posterior_samples = self.prepare_posterior_samples()

        if num_prior_samples is not None:
            if posterior_samples is not None:
                raise ValueError("Cannot provide posterior samples when using prior. ")

        return_sites = [
            "log_prob_mass_1_s_mass_ratio",
            "log_prob_redshift",
            "log_prob_redshift_skyposition",
            "ratio_luminosity_distance_gw_em",
        ]
        if batch_size is None:
            posterior_samples_batch = [posterior_samples]
        else:
            num_posterior_samples = posterior_samples[
                list(posterior_samples.keys())[0]
            ].shape[1]  # first axis is chain dim
            num_batches = (num_posterior_samples + batch_size - 1) // batch_size
            idx_pairs = [
                (i * batch_size, min((i + 1) * batch_size, num_posterior_samples))
                for i in range(num_batches)
            ]
            posterior_samples_batch = [
                jax.tree.map(lambda x: x[:, i:ii], posterior_samples)
                for i, ii in idx_pairs
            ]

        list_of_samples = []
        cpu_device = jax.devices("cpu")[0]
        for i, batch in enumerate(posterior_samples_batch):
            print(
                f"Computing predictive samples for batch {i + 1} / {len(posterior_samples_batch)}. "
            )

            predictive = numpyro.infer.Predictive(
                self.model,
                batch,
                infer_discrete=False,
                return_sites=return_sites,
                batch_ndims=2,
                num_samples=num_prior_samples,
            )
            samples_batch = predictive(
                jax.random.key(0),
                analysis=self,
                **self.data_kwargs,
                use_value_for_Delta=False,  # otherwise conditioning on parameters that have delta priors won't work
                grid_ref=grid_ref,
            )
            samples_batch_cpu = jax.device_put(samples_batch, cpu_device)
            list_of_samples.append(samples_batch_cpu)
        print("Obtained predictive samples. ")

        # concatenate along posterior sample dimension
        samples = jax.tree.map(
            lambda *arrs: jnp.concatenate(arrs, axis=1), *list_of_samples
        )

        return grid_ref, flatten_dict_along_chain_dim(samples)


def load_cosmopyro_results_no_tensors(
    settings_path, model=None, skip_diverging_samples=False
):

    result = Result(model, settings_path=settings_path)
    result.load_inf_data(skip_tensors=True)
    result.load_hypersamples(skip_diverging_samples=skip_diverging_samples)

    return result.hypersamples_df, result.kwargs_analysis
