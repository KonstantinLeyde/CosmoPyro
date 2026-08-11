Quickstart
==========

A CosmoPyro analysis has four steps:

1. **Prepare data** -- posterior samples and injections in HDF5 format
2. **Configure** -- a YAML file specifying the model, priors, and binning
3. **Run** -- launch MCMC from the command line or a Python script
4. **Analyze** -- inspect the posterior samples

This page walks through each step briefly and points to the detailed pages.


1. Prepare data
----------------

You need two HDF5 files:

- **Posterior samples** with shape ``(n_events, n_posterior_samples)``
- **Injections** with shape ``(n_injections,)``

For testing, use the :doc:`simulated O4-like data <simulated_data>` we provide,
or generate your own following that page's instructions.

For real data, see :doc:`running/data_preparation` on how to convert
pandas DataFrames into the required format.


2. Configure
-------------

Create a YAML configuration file specifying the mass model, priors, binning,
and sampler settings.  Example configurations are in the ``examples/configs/``
directory:

- ``examples/configs/powerlaw_2peaks_gwtc5.yaml`` -- multi-peak parametrized model
- ``examples/configs/gp1d_gwtc5.yaml`` -- 1-D Gaussian process for :math:`m_{1,s}` + parametrized :math:`q`
- ``examples/configs/gp2d_gwtc5.yaml`` -- 2-D Gaussian process model in :math:`(\log M, \delta)`

For modified gravity there is no shipped configuration; see
:doc:`methodology/modified_gravity_models` for the ``cosmology_model_name``
values and the priors each one needs.

You can also use the :doc:`interactive configuration builder <running/kwargs_builder>`
to generate a YAML file with a web form.


3. Run
-------

.. code-block:: bash

   python run_analysis.py \
       --job_id my_first_run \
       --path_kwargs examples/configs/powerlaw_2peaks_gwtc5.yaml \
       --path_posterior_samples data/posterior_samples.hdf5 \
       --path_injections data/injections.hdf5

See :doc:`running/launching_analysis` for the full ``run_analysis.py`` script,
all command-line options, and GPU execution.


4. Analyze
-----------

Results are saved to the ``results_path`` specified in the YAML.
See :doc:`running/analyzing_results` for how to load posteriors, make
corner plots, and run convergence diagnostics.


Next steps
----------

- :doc:`methodology/parametrized_mass_models` -- understand the parametrized models
- :doc:`methodology/gaussian_process_mass_models` -- the non-parametric GP alternative
- :doc:`running/sky_density` -- incorporate galaxy catalog information
