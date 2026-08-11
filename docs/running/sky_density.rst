Sky Position Density
====================

CosmoPyro supports a 3-D sky-position prior :math:`p(z, \hat\Omega)` that
modulates the redshift distribution across the sky.  Internally the sky is
discretized using a **HEALPix** grid, so the density is stored as a 2-D array
with shape ``(n_redshift_bins, n_pixels)``.

There are two ways to provide a sky map:

1. **From a galaxy catalog** using ``build_skymap`` (recommended).
2. **From a custom density array** saved as a ``SimpleNamespace``.


Option 1: Build from a galaxy catalog
--------------------------------------

If you have a galaxy catalog with redshifts and sky positions, use
``build_skymap`` to histogram the galaxies into the HEALPix grid:

.. literalinclude:: ../examples/build_skymap_from_catalog.py
   :language: python

The resulting ``SimpleNamespace`` contains:

- ``prob_skyposition_zhp``: normalized density, shape ``(n_z, npix)``
- ``z_edges``: redshift bin edges, shape ``(n_z + 1,)``
- ``nside``: HEALPix resolution parameter

If ``host_log_prob`` is present on the catalog, it is used as weights
(exponentiated) when histogramming.  Otherwise, each galaxy counts equally.

Each redshift slice is independently normalized to sum to 1.  Empty slices
(no galaxies) are filled with a uniform distribution (each pixel set to 1
before normalization) so that every row (i.e. constant redshift) sums (over skyposition) to the same constant.


Option 2: Provide a custom density
------------------------------------

If you already have a sky-position density (e.g. from an external galaxy
catalog pipeline), save it as a ``SimpleNamespace`` with the required fields:

.. literalinclude:: ../examples/custom_density_skymap.py
   :language: python


.. important::

   The ``z_edges`` in the sky map **must exactly match** the redshift bin
   edges used in the analysis (defined by the ``HealPixDiscretization3D``).
   CosmoPyro validates this at runtime and will raise a ``ValueError`` if
   there is a mismatch.


Validation
----------

When a sky map is loaded, CosmoPyro automatically infers ``nside`` from the
second dimension of ``prob_skyposition_zhp`` (``npix = 12 * nside^2``).
You do **not** need to specify ``nside`` in the configuration — it is set to 1
(no sky localisation) when no sky map is provided.

The following are checked at load time:

1. **Number of pixels** (last dimension) equals ``12 * nside^2``.
2. **Redshift edges** match the analysis binning within tolerance (``atol=1e-6``).
3. **Row sums are constant** — ``prob_skyposition_zhp.sum(axis=-1)`` must be
   the same value for every redshift bin (tolerance ``1e-6``). This ensures
   the per-redshift normalization is consistent across slices. If you provide
   a custom sky map, make sure each row sums to the same constant before
   passing it to the analysis.


API reference
-------------

.. autofunction:: cosmopyro.data.build_skymap.build_skymap

.. autofunction:: cosmopyro.data.build_skymap.validate_skymap
