Modified Gravity Models
=======================

CosmoPyro supports several extensions of flat :math:`\Lambda\text{CDM}` that
modify the GW luminosity distance :math:`d_L^{\mathrm{GW}}(z)`.  All models
inherit the standard EM distance from ``FlatLambdaCDM`` and add a
redshift-dependent correction.

Set ``cosmology_model_name`` in the YAML configuration to select a model.
The standard ``FlatLambdaCDM`` requires no extra priors.

The standard ``examples/analyses/run_analysis.py`` path currently accepts
``FlatLambdaCDM``, ``FlatLambdaCDM_GW_distance_cosine``,
``FlatLambdaCDM_GW_distance_gp_integrated``, and
``FlatLambdaCDM_GW_distance_cM``.  The lower-level cosmology factory also
contains ``FlatLambdaCDM_gp_integrated_dLGW``, but that model is not yet wired
through the parameter-completion helper used by ``model_evaluate_p_theta``.


FlatLambdaCDM_GW_distance_gp_integrated
-----------------------------------------

Models the distance **ratio** :math:`\Xi(z) = d_L^{\mathrm{GW}}(z) /
d_L^{\mathrm{EM}}(z)` with a Gaussian process.

A 1-D Gaussian random field in redshift produces ratio nodes; these are
interpolated to give a smooth, flexible :math:`\Xi(z)`.  The GP prior
regularises the ratio toward unity.

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Name
     - Description
   * - ``ratio_gaussian_whitened_field``
     - Whitened noise, shape ``(n_basis,)``.  Prior: :math:`\mathcal{N}(0,1)`.
   * - ``ratio_power_spectrum_amplitude``
     - Controls GP smoothness / amplitude.

**Example YAML:**

.. code-block:: yaml

   cosmology_model_name: FlatLambdaCDM_GW_distance_gp_integrated

   kwargs_priors:
     cosmology:
       h: {dist_type: Uniform, min: 0.3, max: 1.4}
       Omega_m: {dist_type: Delta, value: 0.3}
     modified_ratio:
       ratio_gaussian_whitened_field:
         dist_type: Normal
         loc: 0.0
         scale: 1.0
         shape: [100]
       ratio_power_spectrum_amplitude:
         dist_type: Delta
         value: 0.001


FlatLambdaCDM_GW_distance_cosine
----------------------------------

Expands the ratio as a sum of cosine basis functions:

.. math::

   \Xi(z) = 1 + T(z)\,\sum_{n=1}^{N} \alpha_n \cos\!\bigl(n\pi z / z_{\max} + \varphi_n\bigr)

where :math:`T(z)` is a smooth transition that suppresses the correction at
:math:`z \to 0`.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - ``alphas``
     - Cosine amplitudes, shape ``(N,)``.  Prior: :math:`\mathcal{N}(0, \sigma)`.
   * - ``phases``
     - Cosine phases, shape ``(N,)``.  Prior: Uniform on :math:`[0, 2\pi)`.
   * - ``zmax_b``
     - Maximum redshift of the basis
   * - ``z_tr``
     - Transition redshift for :math:`T(z)`

**Example YAML:**

.. code-block:: yaml

   cosmology_model_name: FlatLambdaCDM_GW_distance_cosine

   kwargs_priors:
     cosmology:
       h: {dist_type: Uniform, min: 0.4, max: 0.9}
       Omega_m: {dist_type: Delta, value: 0.3}
     modified_ratio:
       alphas: {dist_type: Normal, loc: 0.0, scale: 0.04, shape: [15]}
       phases: {dist_type: Uniform, min: 0.0, max: 6.2831853, shape: [15]}
       zmax_b: {dist_type: Delta, value: 1.0}
       z_tr: {dist_type: LogUniform, min: 0.001, max: 1.0}


FlatLambdaCDM_GW_distance_cM
------------------------------

The :math:`c_M` parametrisation models a constant running of the
effective Planck mass:

.. math::

   \Xi(z) = \exp\!\Bigl(\frac{c_M}{2} \int_0^z
     \frac{\Omega_\Lambda(z')}{\Omega_{\Lambda,0}\,(1+z')}\,dz'\Bigr)

A single free parameter :math:`c_M = 0` recovers GR.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - ``cM``
     - Running of the Planck mass

**Example YAML:**

.. code-block:: yaml

   cosmology_model_name: FlatLambdaCDM_GW_distance_cM

   kwargs_priors:
     cosmology:
       h: {dist_type: Uniform, min: 0.4, max: 0.9}
       Omega_m: {dist_type: Delta, value: 0.3}
     modified_ratio:
       cM: {dist_type: Uniform, min: -10.0, max: 10.0}


FlatLambdaCDM_gp_integrated_dLGW
----------------------------------

Instead of modelling the ratio, this model directly parametrises
:math:`d_L^{\mathrm{GW}}(z)` with GP-interpolated nodes.

.. warning::

   This class is available through the lower-level cosmology factory, but it is
   not currently compatible with the standard ``model_evaluate_p_theta`` /
   ``run_analysis.py`` analysis path because the modified-gravity parameter
   completion helper does not yet handle ``modified_luminosity_distance_GW``.

.. list-table::
   :widths: 55 45
   :header-rows: 1

   * - Name
     - Description
   * - ``luminosity_distance_GW_gaussian_whitened_field``
     - Whitened noise, shape ``(n_basis,)``
   * - ``luminosity_distance_GW_amplitude_at_z_max``
     - Value of :math:`d_L^{\mathrm{GW}}` at :math:`z_{\max}`
   * - ``luminosity_distance_GW_power_spectrum_amplitude``
     - GP smoothness / amplitude

**Example YAML:**

.. code-block:: yaml

   cosmology_model_name: FlatLambdaCDM_gp_integrated_dLGW

   kwargs_priors:
     cosmology:
       h: {dist_type: Uniform, min: 0.4, max: 0.9}
       Omega_m: {dist_type: Delta, value: 0.3}
     modified_luminosity_distance_GW:
       luminosity_distance_GW_gaussian_whitened_field:
         dist_type: Normal
         loc: 0.0
         scale: 1
         shape: [30]
       luminosity_distance_GW_amplitude_at_z_max:
         dist_type: Uniform
         min: 10000.0
         max: 50000.0
       luminosity_distance_GW_power_spectrum_amplitude:
         dist_type: Delta
         value: 2000.0
