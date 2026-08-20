Gaussian Process Mass Models
============================

CosmoPyro supports non-parametric mass distributions modeled as
**Gaussian Random Fields** (GRFs).  These provide flexible, data-driven
estimates of the mass distribution without committing to a functional form.

Three variants are available:

- **1-D GP** (``fourier_gp_1D``) — GRF on :math:`m_{1,s}`, combined with a parametrized :math:`p(q \mid m_{1,s})`
- **2-D GP in** :math:`(\log M, \delta)` (``fourier_gp_2D_logMdelta``) — joint GRF with exchange symmetry
- **2-D GP in** :math:`(m_{1,s}, q)` (``fourier_gp_2D_m1sq``) — joint GRF directly on physical coordinates

In each case, the free parameters are **whitened spatial white-noise fields**
with standard-normal priors.  The implementation FFTs those fields and colors
the Fourier modes with the chosen power spectrum, which controls smoothness.


.. tab-set::

   .. tab-item:: fourier_gp_1D

      The 1-D model places a GRF on the ``mass_1_s`` coordinate.

      **Power spectrum:**

      .. math::

         P(k) = A \, \frac{(k/k_c)^2}{1 + (k/k_c)^6}

      where :math:`A` is the amplitude and :math:`k_c` the cutoff wavenumber.

      **Construction:**

      1. Draw whitened spatial noise :math:`\xi_i \sim \mathcal{N}(0, 1)`.
      2. FFT the white-noise field and color it:
         :math:`\hat\phi(k) = \sqrt{P(k)} \cdot \hat\xi(k)`.
      3. Inverse FFT to get the real-space field :math:`\phi(m_{1,s})`.
      4. Apply a mass window :math:`W(m)` (smooth Heaviside at ``mass_min`` / ``mass_max``).
      5. Normalize to a probability density.

      The 1-D GP is combined with a parametrized mass-ratio model (running power
      law), so the full model is :math:`p(m_{1,s}) \cdot p(q \mid m_{1,s})`.

      **Parameters:**

      .. list-table::
         :widths: 40 60
         :header-rows: 1

         * - Name
           - Description
         * - ``gaussian_F_whitened_spatial``
           - Whitened noise vector, shape ``(n_bins,)``.  Prior: :math:`\mathcal{N}(0, 1)`.
         * - ``power_spectrum_amplitude``
           - Overall amplitude :math:`A`
         * - ``power_spectrum_cutoff``
           - Cutoff wavenumber :math:`k_c`
         * - ``mass_min``, ``mass_max``
           - Physical mass bounds for the window function

      **Prior draws:**

      .. literalinclude:: ../examples/gp_1d_prior_draws.py
         :language: python

   .. tab-item:: fourier_gp_2D_logMdelta

      The 2-D model operates in transformed coordinates:

      .. math::

         \log M = \log(m_{1,s} + m_{2,s}), \qquad
         \delta = -\log q = \log(m_{1,s}/m_{2,s})

      where :math:`\log M` is the log total source-frame mass and
      :math:`\delta \ge 0` enforces :math:`m_1 \ge m_2`.

      **Power spectrum:**

      .. math::

         P(k_{\log M}, k_\delta) = A \,
           \frac{\kappa}{(1 + \kappa^2)^3}, \qquad
           \kappa = \frac{\sqrt{k_{\log M}^2 + (r\,k_\delta)^2}}{k_c}

      where :math:`r` is a relative scale parameter allowing different smoothness
      along the two axes.

      **Exchange symmetry:**

      The physical requirement :math:`p(m_1, m_2) = p(m_2, m_1)` translates to
      :math:`\phi(\log M, \delta) = \phi(\log M, -\delta)`.  This is enforced
      **exactly** by mirroring the whitened noise before the FFT.  Only the
      :math:`\delta \ge 0` half is kept after the FFT.

      The joint density is factorized into :math:`p(\log M)` and
      :math:`p(\delta \mid \log M)` for use in the likelihood.

      **Coordinate transforms:**

      .. math::

         m_{1,s} = \frac{e^{\log M}}{1 + e^{-\delta}}, \qquad
         m_{2,s} = \frac{e^{\log M}}{1 + e^{\delta}}

      The implementation adds the coordinate-transform Jacobian when building
      the grid density in :math:`(\log M, \delta)` before factorizing it into
      conditionals for the likelihood.

      **Parameters:**

      .. list-table::
         :widths: 55 45
         :header-rows: 1

         * - Name
           - Description
         * - ``gaussian_F_whitened_spatial``
           - Whitened noise, shape ``(n_logM, n_delta_half)``.
         * - ``power_spectrum_amplitude``
           - Amplitude :math:`A`
         * - ``power_spectrum_cutoff``
           - Cutoff :math:`k_c`
         * - ``power_spectrum_relative_scale_...``
           - Relative scale :math:`r` between axes
         * - ``mass_min``, ``mass_max``
           - Physical mass bounds
         * - ``sigma_low_fractional``, ``sigma_high_fractional``
           - Window slope parameters
         * - ``power_law_reference_mass_1_s``
           - Exponent of the :math:`m_{1,s}^{\alpha}` base-measure factor
             (set to 0 for a flat base measure)
         * - ``power_law_reference_mass_ratio``
           - Exponent of the :math:`q^{\beta}` base-measure factor
              (set to 0 for a flat base measure)

      **Prior draws:**

      .. literalinclude:: ../examples/gp_2d_prior_draws.py
         :language: python

   .. tab-item:: fourier_gp_2D_m1sq

      This variant places the 2-D GRF directly on the :math:`(m_{1,s}, q)` grid.
      Since :math:`q \in (0, 1]` already enforces :math:`m_1 \ge m_2`, no exchange
      symmetry mirroring is needed — the white noise is used directly.

      The power spectrum, window function, and normalization work identically to
      the logMdelta variant, just on different coordinates.

      **Key difference from logMdelta:** no additional Jacobian is needed in the
      likelihood, since the model is already in :math:`(m_{1,s}, q)` coordinates.

      **Parameters:** same as logMdelta, but with
      ``power_spectrum_relative_scale_mass_1_s_to_mass_ratio`` instead of
      ``power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio``.

      **Prior draws:**

      .. literalinclude:: ../examples/gp_2d_m1sq_prior_draws.py
         :language: python


.. seealso::

   Complete, runnable configurations for these models are collected under
   :ref:`example-configurations` on the
   :doc:`../running/launching_analysis` page.

   To experiment with the underlying ``RealField`` class on its own -- drawing
   fields from an arbitrary power spectrum, without any of the mass-model
   machinery -- see :doc:`../tutorials/drawing_random_fields`.


API reference
-------------

.. autofunction:: cosmopyro.distributions.mass_distributions_gaussian_process.construct_log_prob_nn_whitened_field_1D

.. autofunction:: cosmopyro.distributions.mass_distributions_gaussian_process.construct_prob_nn_whitened_field_2D_logMdelta

.. autofunction:: cosmopyro.distributions.mass_distributions_gaussian_process.construct_prob_nn_whitened_field_2D_m1sq
