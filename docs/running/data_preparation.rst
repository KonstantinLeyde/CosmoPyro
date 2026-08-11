Data Preparation
================

CosmoPyro expects input data as ``SimpleNamespace`` objects stored in HDF5
files.  This page shows how to construct the required data structures from
pandas DataFrames and save them using the built-in I/O utilities.


Required data structures
------------------------

The analysis requires two datasets:

**Posterior samples** (``data.samples``):
   A collection of GW events, each with multiple posterior samples from
   parameter estimation.

   Required attributes (each with shape ``(n_events, n_posterior_samples)``):

   - ``mass_1_d`` -- detector-frame primary mass
   - ``mass_ratio`` -- mass ratio :math:`q = m_2/m_1`
   - ``luminosity_distance`` -- luminosity distance (Mpc)
   - ``prior_masses_d_dL`` -- PE prior :math:`p(m_{1,d}, m_{2,d}, d_L)`
   - ``healpix_idx`` -- sky pixel index.  If omitted, ``ra`` and ``dec`` can be
     provided and CosmoPyro will compute ``healpix_idx`` using the analysis
     ``nside``.  If neither is present, a random sky pixel is assigned when
     ``nside`` is available.

**Injections** (``data.injections``):
   A set of simulated signals used to estimate the selection function.

   Required attributes (each 1-D with shape ``(n_injections,)``):

   - ``mass_1_d``, ``mass_ratio``, ``luminosity_distance``
   - ``prior_masses_d_dL``.  Alternatively, store
     ``log_prob_mass_1_d_mass_2_d_luminosity_distance`` and CosmoPyro will
     derive ``prior_masses_d_dL`` while loading.
   - ``healpix_idx``.  As for posterior samples, ``ra`` and ``dec`` may be
     provided instead, or random pixels are assigned when ``nside`` is known.
   - ``num_events`` -- scalar, total number of signals generated **before**
     applying the detection threshold (not the number of detected injections)


Minimal working example
------------------------

The most common starting point is a **list of DataFrames**, one per event,
each containing posterior samples:

.. code-block:: python

   import jax.numpy as jnp
   from types import SimpleNamespace
   from cosmopyro.data.data_utils import save_namespace_to_hdf5

   # ---- 1. Build posterior samples from a list of DataFrames ----
   # event_dfs: list of pd.DataFrame, one per event
   # Each DataFrame has columns: mass_1_d, mass_ratio, luminosity_distance, prior

   # Truncate to the same number of samples per event
   n_samples = min(len(df) for df in event_dfs)

   def stack_column(dfs, col, n_samples):
       return jnp.array([df[col].values[:n_samples] for df in dfs])

   samples = SimpleNamespace(
       mass_1_d=stack_column(event_dfs, "mass_1_d", n_samples),
       mass_ratio=stack_column(event_dfs, "mass_ratio", n_samples),
       luminosity_distance=stack_column(event_dfs, "luminosity_distance", n_samples),
       prior_masses_d_dL=stack_column(event_dfs, "prior", n_samples),
       healpix_idx=stack_column(event_dfs, "healpix_idx", n_samples),
   )
   # samples.mass_1_d.shape == (n_events, n_samples)

   # ---- 2. Build injections from a single DataFrame ----
   # IMPORTANT: num_events is the total number of signals that were
   # generated *before* applying the detection threshold, NOT the
   # number of rows in df_inj (which only contains detected injections).
   # This is needed to correctly estimate the selection function.
   num_total_generated = 5_000_000  # <-- replace with your actual value

   injections = SimpleNamespace(
       mass_1_d=jnp.array(df_inj["mass_1_d"].values),
       mass_ratio=jnp.array(df_inj["mass_ratio"].values),
       luminosity_distance=jnp.array(df_inj["luminosity_distance"].values),
       prior_masses_d_dL=jnp.array(df_inj["prior"].values),
       healpix_idx=jnp.array(df_inj["healpix_idx"].values),
       num_events=jnp.array(num_total_generated),
   )

   # ---- 3. Save ----
   save_namespace_to_hdf5(samples, "posterior_samples.hdf5")
   save_namespace_to_hdf5(injections, "injections.hdf5")

   # ---- 4. Verify by loading back ----
   from cosmopyro.data.data_utils import load_hdf5_to_namespace

   loaded = load_hdf5_to_namespace("posterior_samples.hdf5")
   print(loaded.mass_1_d.shape)  # (n_events, n_samples)


Using ``save_namespace_to_hdf5``
---------------------------------

The function handles:

- JAX and NumPy arrays (compressed with gzip)
- Nested ``SimpleNamespace`` and ``dict`` structures (mapped to HDF5 groups)
- Scalars (stored as 0-D arrays)

.. code-block:: python

   from cosmopyro.data.data_utils import save_namespace_to_hdf5

   save_namespace_to_hdf5(
       data,                 # SimpleNamespace or dict
       "output.hdf5",        # file path
       group="/",            # HDF5 group (default: root)
       overwrite=False,      # set True to overwrite existing file
   )

To overwrite an existing file:

.. code-block:: python

   save_namespace_to_hdf5(data, "output.hdf5", overwrite=True)

.. warning::

   By default, ``save_namespace_to_hdf5`` raises an error if the file already
   exists.  Pass ``overwrite=True`` to replace it.


.. seealso::

   To generate synthetic catalogs for testing, see :doc:`/simulated_data`.


API reference
-------------

.. autofunction:: cosmopyro.data.data_utils.save_namespace_to_hdf5

.. autofunction:: cosmopyro.data.data_utils.load_hdf5_to_namespace
