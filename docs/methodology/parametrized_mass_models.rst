Parametrized Mass Models
========================

CosmoPyro includes three parametrized models for the primary-mass distribution
:math:`p(m_{1,s})` and two for the conditional mass-ratio distribution
:math:`p(q \mid m_{1,s})`. All of them have a smooth low-mass turn-on
:math:`S(m)`, and the turn-on is part of the model name (see below).

All models are evaluated on a discrete grid (bin centres) and then wrapped
in an ``InterpolatedConditional1D`` for likelihood evaluation.

For a non-parametric alternative, see
:doc:`gaussian_process_mass_models` (1-D and 2-D GP models).

.. _low-mass-smoothing:

Low-mass smoothing :math:`S(m)`
-------------------------------

The turn-on :math:`S(m \mid m_\mathrm{min}, \delta_m)` multiplies :math:`p(m_{1,s})`, and
:math:`S(m_2 \mid m_\mathrm{min}, \delta_{m_2})` with :math:`m_2 = q\, m_{1,s}` multiplies
:math:`p(q \mid m_{1,s})`. Two turn-ons are available, and each parametric model is used under
the name ``<model>_smoothing_<smoothing>``:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Model
     - LVK turn-on
     - Gaussian turn-on
   * - ``power_law_peak``
     - ``power_law_peak_smoothing_lvk``
     - ``power_law_peak_smoothing_gaussian``
   * - ``power_law_peak2``
     - ``power_law_peak2_smoothing_lvk``
     - ``power_law_peak2_smoothing_gaussian``
   * - ``power_law_peak2_partial_windowed``
     - ``power_law_peak2_partial_windowed_smoothing_lvk``
     - ``power_law_peak2_partial_windowed_smoothing_gaussian``
   * - ``mass_ratio_running_power_law_in_log``
     - ``mass_ratio_running_power_law_in_log_smoothing_lvk``
     - ``mass_ratio_running_power_law_in_log_smoothing_gaussian``
   * - ``mass_ratio_truncated_gaussian``
     - ``mass_ratio_truncated_gaussian_smoothing_lvk``
     - ``mass_ratio_truncated_gaussian_smoothing_gaussian``

The ``mass_1_s`` and ``mass_ratio`` models must use the same turn-on; mixing them raises an
error. ``fourier_gp_1D`` has no turn-on and can be combined with either mass-ratio variant.
Configurations and results written before the turn-on was part of the name (e.g.
``power_law_peak2_partial_windowed``) are read as ``_smoothing_gaussian``, with a warning.

``lvk``
   The LVK turn-on (`Talbot & Thrane 2018 <https://arxiv.org/abs/1801.02699>`_), as in
   ``gwpopulation`` and ``icarogw``:

   .. math::

      S(m) = \begin{cases}
         S_\mathrm{floor} & m \le m_\mathrm{min} \\
         \max\Bigl\{\Bigl[1 + \exp\Bigl(\frac{\delta_m}{m'} + \frac{\delta_m}{m' - \delta_m}\Bigr)\Bigr]^{-1},\
            S_\mathrm{floor}\Bigr\} & m_\mathrm{min} < m < m_\mathrm{min} + \delta_m \\
         1 & m \ge m_\mathrm{min} + \delta_m
      \end{cases},
      \qquad m' = m - m_\mathrm{min}.

   The exact LVK turn-on is zero below :math:`m_\mathrm{min}`. Here it is floored at
   :math:`S_\mathrm{floor}` = ``LVK_SMOOTHING_FLOOR`` = :math:`10^{-30}`, so that the grid
   log-densities and their gradients stay finite; the floor is negligible against the rest of
   the model. :math:`S(m_\mathrm{min} + \delta_m / 2) = 1/2`.

``gaussian``
   A Gaussian fall-off below a threshold, calibrated to roughly follow the LVK turn-on:

   .. math::

      \log S(m) = -c\, \mathrm{softplus}(t)^2, \qquad
      t = \frac{m_\mathrm{min} + \delta_m + a - m}{b\, \delta_m},

   with :math:`a = -0.1626`, :math:`b = 0.0786`, :math:`c = 0.0250`. It is never exactly zero
   and leaks below :math:`m_\mathrm{min}`, so posteriors on :math:`m_\mathrm{min}` and
   :math:`\delta_m` differ from those with the LVK turn-on and are not directly comparable
   with ``gwpopulation`` / ``icarogw`` results.


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
           - Minimum mass (start of the low-mass turn-on)
         * - ``mmax``
           - Maximum mass for the smooth high-mass window
         * - ``mu_g``
           - Gaussian peak mean
         * - ``sigma_g``
           - Gaussian peak width
         * - ``lambda_peak``
           - Fraction in the Gaussian component (0--1)
         * - ``delta_m``
           - Width of the low-mass turn-on

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
         parameters and differs only in where the sharp window on
         :math:`[m_\mathrm{min}, m_\mathrm{max}]` (whose upper edge is
         :math:`H(m_\mathrm{max} - m)`) is applied: to the power-law component alone
         (as in ``icarogw``), leaving the Gaussian peaks unwindowed, whereas
         ``power_law_peak2`` windows the whole mixture (as in
         ``gwpopulation``). The low-mass turn-on :math:`S(m)` multiplies the whole
         mixture in both.  The shipped example configuration
         ``examples/configs/powerlaw_2peaks_gwtc5.yaml`` uses the partially
         windowed variant.


Mass ratio :math:`p(q \mid m_{1,s})`
--------------------------------------

Two conditional mass-ratio distributions are available, chosen with the
``mass_ratio`` key of ``distribution_names``.  Both carry the low-mass turn-on
:math:`S(m_2)`, with :math:`m_2 = q \cdot m_{1,s}`, of the same kind as the primary-mass model
(:ref:`low-mass-smoothing`).

.. tab-set::

   .. tab-item:: mass_ratio_running_power_law_in_log

      A power law whose exponent varies with the primary mass:

      .. math::

         p(q \mid m_{1,s}) \propto q^{\,\beta(m_{1,s})}
           \; S(m_2 \mid m_\mathrm{min}, \delta_{m_2})

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
           - Width :math:`\delta_{m_2}` of the low-mass turn-on in :math:`m_2`.  If omitted,
             the code falls back to ``delta_m`` from ``mass_1_s``.

   .. tab-item:: mass_ratio_truncated_gaussian

      A Gaussian in :math:`q`:

      .. math::

         p(q \mid m_{1,s}) \propto
           \mathcal{N}(q \mid \mu_q, \sigma_q)
           \; S(m_2 \mid m_\mathrm{min}, \delta_{m_2})

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
           - Width :math:`\delta_{m_2}` of the low-mass turn-on in :math:`m_2`, as above


.. seealso::

   Complete, runnable configurations for these models are collected under
   :ref:`example-configurations` on the
   :doc:`../running/launching_analysis` page.


API reference
-------------

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_prob_nn_power_law_peak_1D

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_prob_nn_multipeak_1D

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_running_power_law_prob_mass_ratio_nn

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_prob_nn_truncated_gaussian_mass_ratio

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.log_smoothing_low_mass_lvk

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.log_smoothing_low_mass_gaussian
