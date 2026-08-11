Analyzing Results
=================

After running MCMC, CosmoPyro saves results to the ``results_path`` specified
in your YAML configuration.  This page shows how to load posteriors, make
corner plots, and visualize the inferred mass distribution.


Example script
--------------

The full example is in ``examples/analyses/analyze_results.py``.
Change ``RESULTS_PATH`` and ``SETTINGS_PATH`` at the top to point to your run.

.. literalinclude:: ../../examples/analyses/analyze_results.py
   :language: python


Step-by-step walkthrough
-------------------------

**1. Load the result object**

The ``Result`` class extends ``Analysis`` and knows how to find all output
files from a completed run:

.. code-block:: python

   from cosmopyro.utils.analyses import Result
   from cosmopyro.models.models import model_evaluate_p_theta

   result = Result(model_evaluate_p_theta, settings_path="path/to/kwargs_analysis.yaml")

**2. Load data and MCMC samples**

.. code-block:: python

   data, skymap = result.load_data()
   result.set_data_kwargs(data=data, analysis=result, skymap=skymap)

   result.load_inf_data(skip_tensors=False)
   result.load_hypersamples(skip_diverging_samples=False)

The hyper-parameter posteriors are available as a pandas DataFrame:

.. code-block:: python

   result.hypersamples_df  # columns: h, gamma, kappa, alpha, ...

**3. Corner plot**

.. code-block:: python

   from cosmopyro.utils.plotting import plot_posterior_comparison

   plot_posterior_comparison(
       {"HMC": result.hypersamples_df},
       filepath="corner.png",
   )

**4. Mass distribution**

.. code-block:: python

   grid_ref, samples_postp = result.compute_main_results()

   from cosmopyro.utils.plotting import make_mass_plot, make_mass_plot_s_delta

   # For (m1_s, q) models:
   make_mass_plot(grid_ref, samples_postp, idx=0, filename="mass_dist.png")

   # For (logM, delta) models:
   make_mass_plot_s_delta(grid_ref, samples_postp, idx=0, filename="mass_dist_sd.png")
