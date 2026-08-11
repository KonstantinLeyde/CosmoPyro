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

The conditional mass-ratio distribution is a power law whose exponent varies
with the primary mass:

.. math::

   p(q \mid m_{1,s}) \propto q^{\,\beta(m_{1,s})}
     \; S(m_2 \mid m_\mathrm{min}, \sigma_{m_2})

where

.. math::

   \beta(m_{1,s}) = \beta_0 + \beta_1 \bigl[\log(m_{1,s}) - \log(m_\mathrm{ref})\bigr]

and :math:`m_2 = q \cdot m_{1,s}`.

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
     - Low-mass smoothing scale for :math:`m_2`

Setting ``mass_ratio: mass_ratio_truncated_gaussian`` instead selects a
Gaussian in :math:`q`, truncated to :math:`0 < q < 1` and carrying the same
low-mass smoothing in :math:`m_2`:

.. math::

   p(q \mid m_{1,s}) \propto
     \mathcal{N}(q \mid \mu_q, \sigma_q)
     \; S(m_2 \mid m_\mathrm{min}, \sigma_{m_2})

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


Example YAML configuration
---------------------------

.. code-block:: yaml

   distribution_names:
     mass_1_s: power_law_peak2
     mass_ratio: mass_ratio_running_power_law_in_log

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


API reference
-------------

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_prob_nn_power_law_peak_1D

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_prob_nn_multipeak_1D

.. autofunction:: cosmopyro.distributions.mass_distribution_parametrized.construct_running_power_law_prob_mass_ratio_nn
