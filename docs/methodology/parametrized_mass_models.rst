Parametrized Mass Models
========================

CosmoPyro includes two parametrized models for the primary-mass distribution
:math:`p(m_{1,s})` and a running power-law model for the conditional
mass-ratio distribution :math:`p(q \mid m_{1,s})`.

All models are evaluated on a discrete grid (bin centres) and then wrapped
in an ``InterpolatedConditional1D`` for likelihood evaluation.

For a non-parametric alternative, see
:doc:`gaussian_process_mass_models` (1-D and 2-D GP models).

.. warning::

   The low-mass smoothing function :math:`S(m)` in CosmoPyro uses a
   **Normal CDF approximation** with calibrated shift and scale parameters.
   This differs from the LVK's original implementation
   (`Talbot & Thrane 2018 <https://arxiv.org/abs/1801.02699>`_), which uses
   a sigmoid based on cumulative numerical integration.  The two are
   functionally similar but not identical, so posteriors on
   :math:`\delta_m` are not directly comparable between the two codes.


Primary mass :math:`p(m_{1,s})`
--------------------------------

.. tab-set::

   .. tab-item:: power_law_peak

      A mixture of a truncated power law and a single Gaussian peak:

      .. math::

         p(m) \propto \bigl[(1 - \lambda_\mathrm{peak})\, m^{-\alpha}
           + \lambda_\mathrm{peak}\,
             \mathcal{N}(m \mid \mu_g, \sigma_g)\bigr]
           \; S(m \mid m_\mathrm{min}, \delta_m)\;
           H(m_\mathrm{max} - m)

      .. list-table::
         :widths: 25 75
         :header-rows: 1

         * - Name
           - Description
         * - ``alpha``
           - Power-law exponent
         * - ``mmin``
           - Minimum mass (smoothing onset)
         * - ``mmax``
           - Maximum mass for the smooth high-mass window
         * - ``mu_g``
           - Gaussian peak mean
         * - ``sigma_g``
           - Gaussian peak width
         * - ``lambda_peak``
           - Fraction in the Gaussian component (0--1)
         * - ``delta_m``
           - Low-mass smoothing scale

   .. tab-item:: power_law_peak2

      Extends the above with **two** Gaussian peaks:

      .. math::

         p(m) \propto \bigl[(1 - \lambda_g)\, m^{-\alpha}
           + \lambda_g \bigl(\lambda_{g,\mathrm{low}}\,
             \mathcal{N}(m \mid \mu_{g,\mathrm{low}}, \sigma_{g,\mathrm{low}})
           + (1 - \lambda_{g,\mathrm{low}})\,
             \mathcal{N}(m \mid \mu_{g,\mathrm{high}}, \sigma_{g,\mathrm{high}})\bigr)
         \bigr]\; S(m) \; H(m_\mathrm{max} - m)

      Shares ``alpha``, ``mmin``, ``mmax``, ``delta_m`` with the single-peak model. Additional parameters:

      .. list-table::
         :widths: 25 75
         :header-rows: 1

         * - Name
           - Description
         * - ``lambda_g``
           - Fraction in the combined Gaussian component
         * - ``lambda_g_low``
           - Fraction of the Gaussian component in the lower peak
         * - ``mu_g_low``, ``sigma_g_low``
           - Lower Gaussian mean and width
         * - ``mu_g_high``, ``sigma_g_high``
           - Upper Gaussian mean and width

      .. note::

         ``power_law_peak2_partial_windowed`` takes exactly the same
         parameters and differs only in where the smoothing window
         :math:`S(m)` is applied: it windows the power-law component alone
         (as in ``icarogw``), leaving the Gaussian peaks unwindowed, whereas
         ``power_law_peak2`` windows the whole mixture (as in
         ``gwpopulation``).  The shipped example configuration
         ``examples/configs/powerlaw_2peaks_gwtc5.yaml`` uses the partially
         windowed variant.


Mass ratio :math:`p(q \mid m_{1,s})`
--------------------------------------

Two conditional mass-ratio distributions are available, chosen with the
``mass_ratio`` key of ``distribution_names``.  Both carry the same low-mass
smoothing :math:`S(m_2)`, with :math:`m_2 = q \cdot m_{1,s}`.

.. tab-set::

   .. tab-item:: mass_ratio_running_power_law_in_log

      A power law whose exponent varies with the primary mass:

      .. math::

         p(q \mid m_{1,s}) \propto q^{\,\beta(m_{1,s})}
           \; S(m_2 \mid m_\mathrm{min}, \sigma_{m_2})

      where

      .. math::

         \beta(m_{1,s}) = \beta_0
           + \beta_1 \bigl[\log(m_{1,s}) - \log(m_\mathrm{ref})\bigr]

      Set ``beta_1`` to a ``Delta`` of 0 for a non-running power law.

      .. list-table::
         :widths: 35 65
         :header-rows: 1

         * - Name
           - Description
         * - ``beta_0``
           - Power-law exponent at reference mass
         * - ``beta_1``
           - Running slope (dependence on :math:`\log m_{1,s}`)
         * - ``mass_ratio_running_zero_point``
           - Reference mass :math:`m_\mathrm{ref}`
         * - ``sigma_mass_cutoff_mass_2``
           - Low-mass smoothing scale for :math:`m_2`.  If omitted, the code
             falls back to ``delta_m`` from ``mass_1_s``.

   .. tab-item:: mass_ratio_truncated_gaussian

      A Gaussian in :math:`q`:

      .. math::

         p(q \mid m_{1,s}) \propto
           \mathcal{N}(q \mid \mu_q, \sigma_q)
           \; S(m_2 \mid m_\mathrm{min}, \sigma_{m_2})

      The truncation is not fixed at :math:`0 < q < 1`: the density is
      normalised over the ``mass_ratio`` grid declared in ``bins``, so its
      support is whatever range you configure there
      (:math:`0.03 \le q \le 1` in the shipped configurations).

      .. list-table::
         :widths: 35 65
         :header-rows: 1

         * - Name
           - Description
         * - ``mu_mass_ratio``
           - Mean :math:`\mu_q` of the Gaussian in :math:`q`
         * - ``sigma_mass_ratio``
           - Width :math:`\sigma_q` of the Gaussian in :math:`q`
         * - ``sigma_mass_cutoff_mass_2``
           - Low-mass smoothing scale for :math:`m_2`, as above


.. seealso::

   Complete, runnable configurations for these models are collected under
   :ref:`example-configurations` on the
   :doc:`../running/launching_analysis` page.


.. toctree::
   :maxdepth: 1
   :caption: Related

   conditional_distributions


API reference
-------------

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_prob_nn_power_law_peak_1D

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_prob_nn_multipeak_1D

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_running_power_law_prob_mass_ratio_nn
