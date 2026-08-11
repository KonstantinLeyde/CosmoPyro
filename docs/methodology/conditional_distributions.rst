1D Conditional Distributions on Grids
======================================

CosmoPyro represents probability distributions as **piecewise-linear densities
on a 1-D grid**, optionally conditioned on one or more discrete variables.
This is the workhorse behind all mass and redshift models: the parametrized
models, the Gaussian-process models, and the redshift model all produce a
density array that is wrapped in the same ``InterpolatedConditional1D`` class
before it enters the likelihood.

Mathematical setup
------------------

Consider a target variable :math:`x` (e.g. ``mass_ratio``) and an optional
set of conditioning variables :math:`\mathbf{y}` (e.g. ``mass_1_s``).

Given bin edges :math:`\{e_0, e_1, \dots, e_N\}` for :math:`x`, the bin
centres are

.. math::

   c_i = \tfrac12(e_i + e_{i+1}), \qquad i = 0, \dots, N-1.

The density :math:`p(x \mid \mathbf{y})` is stored as the array of values
:math:`d_i(\mathbf{y}) = p(c_i \mid \mathbf{y})`.  Between centres the density is
**linearly interpolated**; outside the first and last centre (but still within
the bin edges) it is held **constant**.

.. math::

   p(x \mid \mathbf{y}) =
   \begin{cases}
      d_0(\mathbf{y}) & e_0 \le x < c_0, \\[4pt]
      d_i + \dfrac{d_{i+1} - d_i}{c_{i+1} - c_i}(x - c_i)
        & c_i \le x < c_{i+1}, \\[4pt]
      d_{N-1}(\mathbf{y}) & c_{N-1} \le x < e_N, \\[4pt]
      \varepsilon & \text{otherwise (out of bounds)}.
   \end{cases}

The small constant :math:`\varepsilon \approx 10^{-12}` prevents
:math:`\log 0` for samples outside the grid.


Normalization
-------------

The function ``normalize_cond_interpolated_1d`` ensures that

.. math::

   \int_{e_0}^{e_N} p(x \mid \mathbf{y})\,\mathrm{d}x = 1

for every conditioning slice :math:`\mathbf{y}`.  The integral is computed
analytically from the piecewise-linear model:

.. math::

   I(\mathbf{y}) = \underbrace{\tfrac12 w_0\, d_0}_{\text{left constant}}
     + \sum_{i=0}^{N-2}
       \underbrace{\tfrac12(d_i + d_{i+1}) \cdot \tfrac12(w_i + w_{i+1})}_{\text{linear segment}}
     + \underbrace{\tfrac12 w_{N-1}\, d_{N-1}}_{\text{right constant}}

where :math:`w_i = e_{i+1} - e_i`.

The normalized density is :math:`\hat{d}_i = d_i / I`.


Axis ordering convention
------------------------

``InterpolatedConditional1D`` sorts the **conditioning variable** (``y_bins``)
keys alphabetically to define the axis
ordering of the ``cond`` array:

.. code-block:: python

   # cond.shape == (n_x, n_y1_sorted, n_y2_sorted, ...)
   model = InterpolatedConditional1D(
       x_bins={"mass_ratio": q_edges},
       y_bins={"mass_1_s": m1_edges},  # sorted alphabetically
       cond=prob_q_given_m1,           # shape (n_q, n_m1)
   )

This eliminates a common source of bugs: the caller only needs to build
``cond`` with y-axes in alphabetical key order.  At evaluation time, values
are passed as **dicts**, so the axis mapping is always unambiguous:

.. code-block:: python

   log_p = model.log_prob(
       x_vals={"mass_ratio": q_samples},
       y_vals={"mass_1_s": m1_samples},
   )


Usage example
-------------

.. literalinclude:: ../examples/unconditional_distribution.py
   :language: python


Conditional example
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/conditional_distribution.py
   :language: python


API reference
-------------

.. autoclass:: cosmopyro.distributions.grid_distributions.InterpolatedConditional1D
   :members: log_prob, log_mass

.. autofunction:: cosmopyro.distributions.grid_distributions.normalize_cond_interpolated_1d
