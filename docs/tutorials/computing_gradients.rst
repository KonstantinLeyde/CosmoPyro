Computing Gradients
===================

CosmoPyro is built on JAX, which provides automatic differentiation through
all computations — including cosmology, interpolation, and GP evaluations.

Example 1: Simple 2D function
------------------------------

Define a function and compute its gradient with ``jax.grad``.
The gradient arrows point uphill on the log-probability surface:

.. literalinclude:: ../examples/gradient_simple.py
   :language: python

Example 2: Full model likelihood
---------------------------------

The same approach works for the full CosmoPyro likelihood. Here we compute
the log-likelihood and its gradient on a 2D grid in :math:`(h, \gamma)`:

.. literalinclude:: ../examples/gradient_likelihood.py
   :language: python

.. note::

   The first call will be slow due to JIT compilation. Subsequent evaluations
   are fast. If gradients return ``NaN``, verify that all parameters lie
   within their prior support.
