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

You need two HDF5 files: **posterior samples** and **injections** (the latter
1-D, with shape ``(n_injections,)``).

Posterior samples are accepted in either of two layouts:

*Rectangular* -- every key has shape ``(n_events, n_posterior_samples)``, i.e.
the same number of samples for every event.  Simple, but it forces you to
truncate every event down to the shortest one.

*Flat* -- every key is a 1-D array with all events concatenated, alongside a
``num_posterior_samples_per_event`` key holding the group sizes.  This permits a
**variable number of posterior samples per event**, so nothing has to be thrown
away.  See :ref:`flat-format`.

Rectangular input is flattened internally on load, so both layouts converge on
the same representation before sampling -- the flat layout simply lets you keep
every sample you have.

For testing, use the :doc:`simulated O5-like data <simulated_data>` we provide,
or generate your own following that page's instructions.  The
:doc:`GWTC-5 data products <o5_data>` ship in the flat layout.

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

.. card:: ⚙️  Interactive Configuration Builder
   :link: running/kwargs_builder
   :link-type: doc
   :class-card: sd-border-2 sd-shadow-sm
   :class-title: sd-fs-5

   **Don't write the YAML by hand.**  Pick a mass model, set priors and binning
   in a web form, and download a ready-to-run configuration file.

   +++
   Open the builder →


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
