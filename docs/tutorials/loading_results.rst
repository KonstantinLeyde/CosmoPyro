Loading & Inspecting Results
============================

After running an analysis, this tutorial shows how to load the MCMC chains,
plot posterior distributions, and visualize the inferred mass distribution.

Loading chains
--------------

Point ``Result`` at the ``kwargs_analysis.yaml`` in your results folder:

.. literalinclude:: ../examples/loading_results.py
   :language: python
   :start-at: import jax.numpy as jnp
   :end-before: # --- Step 2

Corner plot
-----------

Produce a corner plot of the scalar hyperparameters:

.. literalinclude:: ../examples/loading_results.py
   :language: python
   :start-at: # --- Step 2
   :end-before: # --- Step 3

Mass distribution
-----------------

To visualize the posterior predictive mass distribution, reload with the
full tensor parameters and call ``compute_main_results``:

.. literalinclude:: ../examples/loading_results.py
   :language: python
   :start-at: # --- Step 3
   :end-before: # --- Step 4

Comparing with the true population
-----------------------------------

If you have a reference catalog (e.g. from simulated data), you can overlay
the true population:

.. literalinclude:: ../examples/loading_results.py
   :language: python
   :start-at: # --- Step 4
