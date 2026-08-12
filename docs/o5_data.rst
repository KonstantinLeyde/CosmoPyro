GWTC-5 Data
===========

Pre-processed data products from the LIGO-Virgo-KAGRA GWTC-5 catalog are
available for direct use with CosmoPyro.

Both files are distributed in **flat** format: the posterior samples of all
events are concatenated into single 1-D arrays rather than stored as a
rectangular ``(n_events, n_posterior_samples)`` block.  A
``num_posterior_samples_per_event`` key records how many consecutive samples
belong to each event, so events may have different numbers of samples.  See
:ref:`flat-format` below.


Downloads
---------

All files live under
https://users.flatironinstitute.org/~kleyde/dark_sirens/cosmopyro/gwtc-5/

.. list-table::
   :widths: 28 60 20
   :header-rows: 1

   * - File
     - Description
     - Link
   * - ``gwtc-5-bbh-flat``
     - PE posterior samples for the confident BBH events, flattened across
       events (see :ref:`flat-format`).  Keys ``mass_1_d``, ``mass_ratio``,
       ``luminosity_distance``, ``prior_masses_d_dL``, plus
       ``num_posterior_samples_per_event``.
     - `Download <https://users.flatironinstitute.org/~kleyde/dark_sirens/cosmopyro/gwtc-5/gwtc-5-bbh-flat>`__
   * - ``mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped``
     - Injection set for selection-function estimation (semi-analytic O1/O2
       plus real O3/O4a/O4b sensitivities, polar spins, clipped).  Same column
       keys plus the scalar ``num_events``.  About 47 MB.
     - `Download <https://users.flatironinstitute.org/~kleyde/dark_sirens/cosmopyro/gwtc-5/mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped>`__

Both are HDF5 files despite carrying no ``.hdf5`` extension.  To fetch them
into a local ``data/`` directory:

.. code-block:: bash

   base=https://users.flatironinstitute.org/~kleyde/dark_sirens/cosmopyro/gwtc-5
   mkdir -p data

   curl -L -o data/gwtc-5-bbh-flat "$base/gwtc-5-bbh-flat"
   curl -L -o data/mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped \
       "$base/mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped"


Usage
-----

Once downloaded, run the analysis:

.. code-block:: bash

   python run_analysis.py \
       --job_id gwtc5_plp2 \
       --path_kwargs examples/configs/powerlaw_2peaks_gwtc5.yaml \
       --path_posterior_samples data/gwtc-5-bbh-flat \
       --path_injections data/mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped

Or load and inspect from Python:

.. code-block:: python

   from cosmopyro.data.data_utils import load_hdf5_to_namespace

   samples = load_hdf5_to_namespace("data/gwtc-5-bbh-flat")
   injections = load_hdf5_to_namespace(
       "data/mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped"
   )

   n_per_event = samples.num_posterior_samples_per_event

   print(f"Events: {n_per_event.shape[0]}")
   print(f"Posterior samples, total: {samples.mass_1_d.shape[0]}")
   print(f"Posterior samples per event: {n_per_event.min()}-{n_per_event.max()}")
   print(f"Injections: {injections.mass_1_d.shape[0]}")

Note that ``samples.mass_1_d`` is 1-D here, so the event count comes from
``num_posterior_samples_per_event`` rather than from the array shape.


.. _flat-format:

Flat format
-----------

In the rectangular format described in :doc:`running/data_preparation`, every
event must contribute the same number of posterior samples, which forces you to
truncate every event down to the smallest one.  The flat format avoids that: all
events are simply concatenated.

**Posterior samples** -- each key is 1-D with shape
``(sum(num_posterior_samples_per_event),)``:

- ``mass_1_d``: detector-frame primary mass (:math:`M_\odot`)
- ``mass_ratio``: mass ratio :math:`q = m_2/m_1`
- ``luminosity_distance``: luminosity distance (Mpc)
- ``prior_masses_d_dL``: PE sampling prior :math:`p(m_{1,d}, m_{2,d}, d_L)`
- ``num_posterior_samples_per_event``: 1-D integer array of length
  ``n_events``, giving the group sizes in the order the samples are stored

**Injections** -- shape ``(n_injections,)``:

- Same columns as above, plus:
- ``num_events``: total number of injections generated (before selection)

``load_posterior_samples_and_injections_from_file`` detects the flat layout
automatically: if ``mass_1_d`` is 1-D and ``num_posterior_samples_per_event`` is
present, the samples are used as-is.  Otherwise a 2-D block is down-selected and
then flattened internally with
:func:`~cosmopyro.data.load_catalogs.flatten_samples_to_stacked`, so the
likelihood always sees the same stacked representation.  When the group sizes are
not all equal, ``posterior_sample_segment_ids`` is derived from them so that
per-event sums can be computed with a segment reduction.

.. warning::

   ``--num_events`` and ``num_posterior_samples`` cannot be combined with
   pre-flattened data -- the loader raises a ``ValueError``, because sub-selecting
   events would invalidate the stored group sizes.  Down-select before flattening
   if you need a smaller dataset.
