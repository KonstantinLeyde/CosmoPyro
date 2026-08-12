Installation
============

Requirements
------------

- **Python 3.11 or later**
- A JAX build for your hardware (CPU by default, see :ref:`gpu-support`)

Everything else is installed automatically. The core dependencies are
`NumPyro <https://num.pyro.ai/en/stable/>`_ and
`JAX <https://docs.jax.dev/en/latest/>`_ for the sampling, plus ``numpy``,
``scipy``-adjacent tooling (``arviz``, ``blackjax``, ``optax``, ``equinox``),
``h5py`` for I/O, ``healpy`` for sky maps, and ``matplotlib``/``corner`` for
plotting.

.. tip::

   We recommend `uv <https://docs.astral.sh/uv/>`_ throughout. It is a drop-in
   replacement for ``pip`` that resolves dependencies significantly faster,
   which is noticeable when installing JAX. Plain ``pip`` works everywhere
   ``uv pip`` is shown below.


Install from PyPI
-----------------

The quickest route, if you only want to *use* CosmoPyro:

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install cosmopyro


Install from source
-------------------

Use this if you want to modify the code, run the test suite, or work through
the example notebooks:

.. code-block:: bash

   git clone https://github.com/KonstantinLeyde/CosmoPyro.git
   cd CosmoPyro

   uv venv --python 3.11
   source .venv/bin/activate

   uv pip install -e .

The ``-e`` (editable) install means changes to the source tree take effect
without reinstalling.


.. _gpu-support:

GPU support
-----------

All numerical work goes through JAX, so running on GPU is a matter of
installing the JAX build that matches your cluster's CUDA version. Do this
*after* installing CosmoPyro, since the base install pulls in CPU-only JAX:

.. code-block:: bash

   pip install --upgrade --force-reinstall "jax[cuda12]"

``--force-reinstall`` is what replaces the CPU wheel already present. See the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for CUDA 11, TPU, or Apple Metal.

.. note::

   CosmoPyro enables 64-bit precision in the example scripts via
   ``jax.config.update("jax_enable_x64", True)``. Keep this on: the likelihood
   sums log-probabilities over millions of samples, and float32 is not enough
   to keep those stable.


Verify the installation
-----------------------

.. code-block:: bash

   python -c "import cosmopyro; print(cosmopyro.__version__)"

Then run the test suite (needs the ``test`` extra):

.. code-block:: bash

   pytest tests/ -q


Building the documentation
--------------------------

.. code-block:: bash

   uv pip install -e ".[docs]"
   cd docs
   make html

The rendered pages land in ``docs/_build/html``; open ``index.html``. Use
``make clean html`` to force a full rebuild.


Next steps
----------

- :doc:`quickstart` for a first run.
- The `examples folder <https://github.com/KonstantinLeyde/CosmoPyro/tree/main/examples>`_
  in the repository, which has its own README.
- The `CosmoPyro-Demo <https://github.com/KonstantinLeyde/CosmoPyro-Demo>`_
  repository for standalone worked examples.
