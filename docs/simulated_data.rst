Simulated O4-like Data
======================

For testing and development, we provide simulated data that mimics an
O4-like observing run.  These datasets have **no uncertainty on sky position**,
making them simpler to work with than real data.


Downloads
---------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - File
     - Description
     - Link
   * - ``posterior_samples_31.hdf5``
     - Synthetic PE posterior samples generated with a known
       population (multi-peak mass model, :math:`H_0 = 70` km/s/Mpc).
       Includes gamma-distributed measurement uncertainties.
     - `Download <https://users.flatironinstitute.org/~kleyde/dark_sirens/cosmopyro/fake_data/o5/posterior_samples_31.hdf5>`__
   * - ``injections_17.hdf5``
     - Corresponding injection set with SNR cut, for
       selection-function estimation.
     - `Download <https://users.flatironinstitute.org/~kleyde/dark_sirens/cosmopyro/fake_data/o5/injections_17.hdf5>`__


Generating your own
--------------------

The simulated data can be reproduced with the scripts in ``examples/make_data/``:

- `produce_catalog.py <https://github.com/konstantinleyde/cosmopyro/blob/main/examples/make_data/produce_catalog.py>`__
  -- generates catalogs with SNR selection and a reference set without SNR cut
- `produce_injection_set.py <https://github.com/konstantinleyde/cosmopyro/blob/main/examples/make_data/produce_injection_set.py>`__
  -- generates large injection sets for selection-function estimation
- `produce_posterior_samples.py <https://github.com/konstantinleyde/cosmopyro/blob/main/examples/make_data/produce_posterior_samples.py>`__
  -- turns a catalog into PE posterior samples by adding measurement uncertainties

To generate a catalog:

.. code-block:: bash

   cd examples/make_data
   python produce_catalog.py --catalog_idx 0

This produces a combined catalog of selected events (SNR > 12) and a
reference catalog with no SNR cut (for plotting).

PE measurement uncertainties are then added by ``produce_posterior_samples.py``
(which wraps ``cosmopyro.data.generate_simple_pe.compute_posterior_samples``):

.. code-block:: bash

   python produce_posterior_samples.py --catalog_idx 0


What is **not** included
-------------------------

These simulated datasets do not include:

- **Sky position uncertainty** -- all events have a single sky location
  (no ``ra``/``dec`` posterior samples)
- **Spin effects** -- the mass model is spin-agnostic
- **Realistic noise curves** -- SNR is computed from an analytic approximation

For analyses that require sky localization, see :doc:`running/sky_density`
on how to incorporate galaxy catalog information.
