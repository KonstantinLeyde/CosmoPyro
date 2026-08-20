Launching an Analysis
=====================

This page shows how to run a CosmoPyro analysis from the command line.
Before launching, you need a YAML configuration file — use the
:doc:`interactive configuration builder <kwargs_builder>` to generate one,
or start from one of the examples in ``examples/configs/``.


Launch script
--------------

A ready-to-use launch script is provided at
``examples/analyses/run_analysis.py``:

.. literalinclude:: ../../examples/analyses/run_analysis.py
   :language: python

Invoke from the terminal:

.. code-block:: bash

   python run_analysis.py \
       --debug false \
       --job_id ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
       --path_posterior_samples /mnt/home/kleyde/ceph/cosmology/dark_siren/data/real_data/gwtc-5-bbh-flat \
       --path_injections /mnt/home/kleyde/ceph/cosmology/dark_siren/data/real_data/injections/mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped \
       --path_kwargs $path_kwargs \
       --sampler nuts


Command-line arguments
-----------------------

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Argument
     - Required
     - Description
   * - ``--job_id``
     - Yes
     - Unique identifier for this run (used in output filenames)
   * - ``--num_events``
     - No
     - Number of GW events to use (default: all available)
   * - ``--num_posterior_samples``
     - No
     - Number of PE posterior samples per event (default: all available)
   * - ``--path_kwargs``
     - No
     - Path to the YAML configuration file
   * - ``--path_posterior_samples``
     - No
     - Path to HDF5 file with PE posterior samples
   * - ``--path_injections``
     - No
     - Path to HDF5 file with injection samples
   * - ``--path_skymap``
     - No
     - Path to HDF5 sky map file
   * - ``--sampler``
     - No
     - Sampler type: ``nuts`` (default) or ``mclmc``
   * - ``--num_injections``
     - No
     - Cap on number of injections to load
   * - ``--debug``
     - No
     - Enable debug mode (``true``/``false``)


YAML configuration overview
-----------------------------

A minimal YAML configuration file looks like:

.. code-block:: yaml

   catalog_metadata:
     healpix_discretization_3d: {}

   # Cosmology model used by examples/analyses/run_analysis.py:
   # FlatLambdaCDM, FlatLambdaCDM_GW_distance_cosine,
   # FlatLambdaCDM_GW_distance_gp_integrated, or FlatLambdaCDM_GW_distance_cM
   cosmology_model_name: FlatLambdaCDM
   cosmology_numerics:
     z_max: 5.0
     z_min: 1.0e-05
     z_steps: 200000

   # Factorized: p(m1_s) * p(q | m1_s)
   distribution_names:
     mass_1_s: power_law_peak2    # or: power_law_peak,
                                  # power_law_peak2_partial_windowed, fourier_gp_1D
     mass_ratio: mass_ratio_running_power_law_in_log
                                  # or: mass_ratio_truncated_gaussian
     redshift: MadauDickinson
   # OR joint: p(logM, delta) or p(m1_s, q)
   # distribution_names:
   #   source_frame_masses: fourier_gp_2D_logMdelta  # or: fourier_gp_2D_m1sq
   #   redshift: MadauDickinson

   kwargs_priors:
     cosmology:
       h: {dist_type: Uniform, min: 0.1, max: 2.0}
       Omega_m: {dist_type: Delta, value: 0.3}
     redshift:
       gamma: {dist_type: Uniform, min: 0.0, max: 5.0}
       kappa: {dist_type: Uniform, min: 0.0, max: 6.0}
       zp: {dist_type: Uniform, min: 0.0, max: 4.0}
     mass_1_s:
       alpha: {dist_type: Uniform, min: 1.5, max: 6.0}
       # ... (see Parametrized Mass Models page)
     mass_ratio:
       beta_0: {dist_type: Uniform, min: -2.0, max: 4.0}
       # ...

   kwargs_sampler:
     num_chains: 1
     num_warmup: 1000
     num_posterior_samples: 1000
     num_posterior_samples_per_batch: 100
     max_tree_depth: 10
     target_accept_prob: 0.8
     forward_mode_differentiation: false  # use forward-mode AD (useful for large latent spaces)
     # SVI initialization
     num_svi_steps: 0          # set >0 to enable SVI initialization
     num_svi_samples: 0        # samples drawn from SVI guide
     guide_type: AutoLowRankMultivariateNormal  # AutoLowRankMultivariateNormal, AutoIAFNormal, AutoBNAFNormal, AutoBNAFNormal_AutoNormal, or AutoLowRankMVN_AutoNormal
     # num_events_svi: 100     # optional: use fewer events for SVI initialization
     # guide_hidden_dims: [128, 128]    # hidden layer sizes for AutoIAFNormal
     # guide_hidden_factors: [10, 10, 10]  # hidden layer factors for AutoBNAFNormal
     # Gibbs sampling (optional)
     # gibbs_sites: [h]                    # params sampled via Gibbs MH instead of NUTS
     # gibbs_mh_step_size: 0.01            # MH proposal step size
     # gibbs_num_mh_steps: 10              # inner MH steps per HMC iteration
     # Advanced
     # desired_energy_variance: 1.0e-9     # for MCLMC sampler
     # start_nuts_from_previous_run_path: null  # path to reuse mass matrix from a previous run

   likelihood_evaluation:
     posterior_samples_batch_size: 500000
     injections_batch_size: 600000
     nonfinite_log_prob_policy: repair  # repair (default) or strict
     save_effective_sample_sizes: false
     # penalty_factor_relative_variance: 1.0  # false/omitted disables the penalty

   bins:
     mass_1_s: {min: 1.0, max: 150.0, num: 400}
     mass_ratio: {min: 0.03, max: 1.0, num: 200}
     redshift: {min: 0.0, max: 5.0, num: 1000}
     mass_grid_interpolation: smooth_log      # or linear_density
     redshift_grid_interpolation: smooth_log  # or linear_density

   results_path: ./results/

   # Added automatically by examples/analyses/run_analysis.py at launch time.
   # Include this block only when loading a saved results kwargs_analysis.yaml
   # or when constructing an Analysis object directly.
   # run_kwargs:
   #   job_id: run_001
   #   path_kwargs: examples/configs/powerlaw_2peaks_gwtc5.yaml
   #   path_posterior_samples: data/posterior_samples.hdf5
   #   path_injections: data/injections.hdf5
   #   path_skymap: null
   #   num_events: null
   #   num_posterior_samples: null
   #   num_injections: null
   #   sampler: nuts
   #   debug: false

The model-specific blocks for each mass model are collected under
:ref:`example-configurations` below, and ready-to-run files live in the
``examples/configs/`` directory.

You can also use the :doc:`interactive configuration builder <kwargs_builder>`
to generate a YAML file.  For details on all available models, see the
:doc:`../methodology/modified_gravity_models` page.

Settings read by the current analysis code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main YAML object is ``kwargs_analysis``.  These top-level keys are read by
``Analysis`` and ``model_evaluate_p_theta``:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Key
     - Used for
   * - ``catalog_metadata``
     - Enables sky/catalog handling.  ``healpix_discretization_3d.nside`` is
       used when present; if a skymap is loaded, ``nside`` is inferred from it.
   * - ``catalog_settings_filename``
     - Optional path to catalog metadata when it is not embedded directly.
   * - ``cosmology_model_name``
     - Selects the cosmology model.  The main analysis path currently supports
       ``FlatLambdaCDM``, ``FlatLambdaCDM_GW_distance_cosine``,
       ``FlatLambdaCDM_GW_distance_gp_integrated``, and
       ``FlatLambdaCDM_GW_distance_cM``.
   * - ``cosmology_numerics``
     - Redshift interpolation settings: ``z_min``, ``z_max``, ``z_steps``.
   * - ``distribution_names``
     - Selects mass and redshift models.  Factorized mass models use
       ``mass_1_s`` and ``mass_ratio``; joint GP models use
       ``source_frame_masses``.  Optional GP keys are
       ``source_frame_masses_power_spectrum`` and
       ``field_to_log_prob_prescription``.
   * - ``kwargs_priors``
     - Prior definitions grouped by parameter block, for example
       ``cosmology``, ``redshift``, ``mass_1_s``, ``mass_ratio``,
       ``source_frame_masses``, or ``modified_ratio``.
   * - ``kwargs_sampler``
     - NUTS/MCLMC/SVI/Gibbs settings listed in the example above.
   * - ``likelihood_evaluation``
     - Batch sizes and likelihood error-handling settings.
   * - ``bins``
     - Grid boundaries for mass/redshift variables and interpolation choices.
   * - ``injection_evaluation``
     - Optional.  Defaults to ``exclude_skyposition``.  Any other value uses
       the same sky-position distribution for injections and posterior samples.
   * - ``results_path``
     - Output root.  Results are written under ``results_path/id_<job_id>/``.
   * - ``run_kwargs``
     - Runtime data paths and CLI options.  The provided launch script writes
       this block into the saved copy of the YAML.

The following keys still appear in some older example files but are not read by
the current runtime: ``kwargs_data``, ``gibbs_mode``,
``redshift_model_name``, and top-level ``mass_ratio_running_zero_point``.


Available prior types
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Type
     - Parameters
   * - ``Delta``
     - Fixed value: ``value``
   * - ``Uniform``
     - Flat: ``min``, ``max``
   * - ``LogUniform``
     - Log-flat: ``min``, ``max``
   * - ``Normal``
     - Gaussian: ``loc``, ``scale`` (optionally ``shape`` for vector params)
   * - ``Dirichlet``
     - Dirichlet: ``concentration``


.. _example-configurations:

Example configurations
-----------------------

The model-specific blocks for every mass model shipped with CosmoPyro,
collected here rather than spread across the model pages.  Each tab is a
fragment: combine it with the ``cosmology_model_name``, ``kwargs_sampler``,
``likelihood_evaluation`` and ``results_path`` blocks from the overview above
to get a complete configuration file.

.. tab-set::

   .. tab-item:: power_law_peak2

      Parametrized multi-peak primary mass with a running power law in
      :math:`q` -- see :doc:`../methodology/parametrized_mass_models`.

      .. code-block:: yaml

         distribution_names:
           mass_1_s: power_law_peak2
           mass_ratio: mass_ratio_running_power_law_in_log
           # or: mass_ratio_truncated_gaussian, which replaces beta_0/beta_1/
           # mass_ratio_running_zero_point below with mu_mass_ratio and
           # sigma_mass_ratio

         kwargs_priors:
           mass_1_s:
             alpha:       {dist_type: Uniform, min: 1.5, max: 6.0}
             mmin:        {dist_type: Uniform, min: 2.0, max: 10.0}
             mmax:        {dist_type: Uniform, min: 50.0, max: 200.0}
             lambda_g:    {dist_type: Uniform, min: 0.0, max: 1.0}
             lambda_g_low:{dist_type: Uniform, min: 0.0, max: 1.0}
             delta_m:     {dist_type: Uniform, min: 0.001, max: 10.0}
             mu_g_low:    {dist_type: Uniform, min: 5.0, max: 15.0}
             sigma_g_low: {dist_type: Uniform, min: 0.4, max: 5.0}
             mu_g_high:   {dist_type: Uniform, min: 15.0, max: 100.0}
             sigma_g_high:{dist_type: Uniform, min: 0.4, max: 10.0}
           mass_ratio:
             beta_0: {dist_type: Uniform, min: -2.0, max: 4.0}
             beta_1: {dist_type: Delta, value: 0.0}
             mass_ratio_running_zero_point: {dist_type: Delta, value: 10.0}

         bins:
           mass_1_s:   {min: 1.0, max: 150.0, num: 400}
           mass_ratio: {min: 0.03, max: 1.0, num: 200}

   .. tab-item:: fourier_gp_1D

      1-D Gaussian process on :math:`m_{1,s}`, combined with a parametrized
      :math:`p(q \mid m_{1,s})` -- see
      :doc:`../methodology/gaussian_process_mass_models`.

      .. code-block:: yaml

         distribution_names:
           mass_1_s: fourier_gp_1D
           mass_ratio: mass_ratio_running_power_law_in_log
           redshift: MadauDickinson

         kwargs_priors:
           mass_1_s:
             gaussian_F_whitened_spatial:
               dist_type: Normal
               loc: 0.0
               scale: 1
               shape: [200]
             mass_min: {dist_type: Uniform, min: 2, max: 10}
             mass_max: {dist_type: Uniform, min: 50, max: 150}
             sigma_low_fractional: {dist_type: Uniform, min: 0.01, max: 0.1}
             sigma_high_fractional: {dist_type: Uniform, min: 0.01, max: 0.1}
             power_spectrum_amplitude: {dist_type: Delta, value: 5.0}
             power_spectrum_cutoff: {dist_type: Delta, value: 5.0}

         bins:
           mass_1_s: {min: 2.0, max: 120.0, num: 200}
           mass_ratio: {min: 0.03, max: 1.0, num: 200}
           redshift: {min: 0.0, max: 5.0, num: 1000}

   .. tab-item:: fourier_gp_2D_logMdelta

      Joint 2-D Gaussian process in :math:`(\log M, \delta)`, with exchange
      symmetry enforced exactly -- see
      :doc:`../methodology/gaussian_process_mass_models`.

      .. code-block:: yaml

         distribution_names:
           source_frame_masses: fourier_gp_2D_logMdelta
           redshift: MadauDickinson

         kwargs_priors:
           source_frame_masses:
             gaussian_F_whitened_spatial:
               dist_type: Normal
               loc: 0.0
               scale: 1
               shape: [120, 120]
             mass_min: {dist_type: Uniform, min: 2, max: 10}
             mass_max: {dist_type: Uniform, min: 30, max: 120}
             sigma_low_fractional: {dist_type: Uniform, min: 0.01, max: 0.2}
             sigma_high_fractional: {dist_type: Uniform, min: 0.01, max: 0.2}
             power_spectrum_amplitude: {dist_type: Delta, value: 0.045}
             power_spectrum_cutoff: {dist_type: Delta, value: 50.0}
             power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio:
               dist_type: Delta
               value: 1.0
             power_law_reference_mass_1_s: {dist_type: Delta, value: -2.0}
             power_law_reference_mass_ratio: {dist_type: Delta, value: 1.5}

         bins:
           log_mass_total_s:    {min: 1.5, max: 6.0, num: 120}
           minus_log_mass_ratio: {min: 0.0, max: 4.0, num: 120}
           redshift:            {min: 0.0, max: 5.0, num: 1000}

   .. tab-item:: fourier_gp_2D_m1sq

      Joint 2-D Gaussian process directly on :math:`(m_{1,s}, q)`, which needs
      no extra Jacobian -- see
      :doc:`../methodology/gaussian_process_mass_models`.

      .. code-block:: yaml

         distribution_names:
           source_frame_masses: fourier_gp_2D_m1sq
           redshift: MadauDickinson

         kwargs_priors:
           source_frame_masses:
             gaussian_F_whitened_spatial:
               dist_type: Normal
               loc: 0.0
               scale: 1
               shape: [120, 120]
             mass_min: {dist_type: Uniform, min: 2, max: 10}
             mass_max: {dist_type: Uniform, min: 30, max: 120}
             sigma_low_fractional: {dist_type: Uniform, min: 0.01, max: 0.2}
             sigma_high_fractional: {dist_type: Uniform, min: 0.01, max: 0.2}
             power_spectrum_amplitude: {dist_type: Delta, value: 0.045}
             power_spectrum_cutoff: {dist_type: Delta, value: 50.0}
             power_spectrum_relative_scale_mass_1_s_to_mass_ratio:
               dist_type: Delta
               value: 1.0

         bins:
           mass_1_s:   {min: 2.0, max: 120.0, num: 120}
           mass_ratio: {min: 0.02, max: 1.0, num: 120}
           redshift:   {min: 0.0, max: 5.0, num: 1000}

Ready-to-run files for these models live in the ``examples/configs/``
directory of the repository, and the
:doc:`interactive configuration builder <kwargs_builder>` will generate one
for any combination of models.
