GWTC-4 Data
===========

Pre-processed data products from the LIGO-Virgo-KAGRA GWTC-4 catalog are
available for direct use with CosmoPyro.


Downloads
---------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - File
     - Description
     - Link
   * - ``gwtc4_posterior_samples.hdf5``
     - PE posterior samples for all confident BBH events,
       formatted as ``(n_events, n_posterior_samples)`` arrays
       with keys ``mass_1_d``, ``mass_ratio``, ``luminosity_distance``,
       ``prior_masses_d_dL``.
     - `Coming soon <#>`__
   * - ``gwtc4_injections.hdf5``
     - Injection set for selection-function estimation,
       with the same column keys plus ``num_events``.
     - `Coming soon <#>`__

.. note::

   Download links will be provided once the data release is finalized.
   Check back or watch the
   `GitHub repository <https://github.com/konstantinleyde/cosmopyro>`__
   for updates.


Usage
-----

Once downloaded, run the analysis:

.. code-block:: bash

   python run_analysis.py \
       --job_id gwtc4_plp2 \
       --path_kwargs examples/configs/powerlaw_2peaks_gwtc5.yaml \
       --path_posterior_samples data/gwtc4_posterior_samples.hdf5 \
       --path_injections data/gwtc4_injections.hdf5

Or load and inspect from Python:

.. code-block:: python

   from cosmopyro.data.data_utils import load_hdf5_to_namespace

   samples = load_hdf5_to_namespace("data/gwtc4_posterior_samples.hdf5")
   injections = load_hdf5_to_namespace("data/gwtc4_injections.hdf5")

   print(f"Events: {samples.mass_1_d.shape[0]}")
   print(f"Posterior samples per event: {samples.mass_1_d.shape[1]}")
   print(f"Injections: {injections.mass_1_d.shape[0]}")


Data format
-----------

Both files follow the standard CosmoPyro HDF5 format (see
:doc:`running/data_preparation`).

**Posterior samples** -- shape ``(n_events, n_posterior_samples)``:

- ``mass_1_d``: detector-frame primary mass (:math:`M_\odot`)
- ``mass_ratio``: mass ratio :math:`q = m_2/m_1`
- ``luminosity_distance``: luminosity distance (Mpc)
- ``prior_masses_d_dL``: PE sampling prior :math:`p(m_{1,d}, m_{2,d}, d_L)`

**Injections** -- shape ``(n_injections,)``:

- Same columns as above, plus:
- ``num_events``: total number of injections generated (before selection)
