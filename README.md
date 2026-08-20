<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/KonstantinLeyde/CosmoPyro/refs/heads/main/images/CosmoPyro_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/KonstantinLeyde/CosmoPyro/refs/heads/main/images/CosmoPyro_light.png">
    <!-- PyPI strips <source> elements, so this fallback is what renders there. -->
    <img alt="CosmoPyro logo" src="https://raw.githubusercontent.com/KonstantinLeyde/CosmoPyro/refs/heads/main/images/CosmoPyro_light.png" width="280">
  </picture>
</div>

# **CosmoPyro**

[![Documentation](https://readthedocs.org/projects/cosmopyro/badge/?version=latest)](https://cosmopyro.readthedocs.io/)

CosmoPyro is a Bayesian framework for reconstructing the gravitational-wave population of compact binary coalescences using Hamiltonian Monte Carlo (HMC), and more specifically the No-U-Turn sampler implemented in [NumPyro](https://num.pyro.ai/en/stable/).

📖 **Full documentation: [cosmopyro.readthedocs.io](https://cosmopyro.readthedocs.io/)** — [installation](https://cosmopyro.readthedocs.io/en/latest/installation.html), a [quickstart](https://cosmopyro.readthedocs.io/en/latest/quickstart.html), the mass and cosmology model reference, and an [interactive configuration builder](https://cosmopyro.readthedocs.io/en/latest/running/kwargs_builder.html).

## Installation Instructions

### Installation from PyPI

If you want to install CosmoPyro as quickly as possible, you can use 

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install cosmopyro
```
Alternatively, if you want to modify the code locally, you can install the code via cloning the git repository. 

### Basic installation from git

To create a new environment and install **CosmoPyro**, follow these steps. We would recommend installing `uv`, since it is significantly faster at resolving dependencies, but otherwise, you can also rely on `pip` alone. 

```bash
python -m venv .venv
source .venv/bin/activate
git clone git@github.com:KonstantinLeyde/CosmoPyro.git
cd CosmoPyro
uv pip install -e .
```

### Making the Environment GPU Compatible
To run on GPUs, you'll need the GPU version of `jax` matching your cluster's CUDA version:

```bash
pip install --upgrade --force-reinstall "jax[cuda12]"
```

## Running

See the [Quickstart](https://cosmopyro.readthedocs.io/en/latest/quickstart.html) for a first run, and [Launching an Analysis](https://cosmopyro.readthedocs.io/en/latest/running/launching_analysis.html) for the full set of command-line options.

For worked examples, see the dedicated [CosmoPyro-Demo](https://github.com/KonstantinLeyde/CosmoPyro-Demo) repository, or the notebooks in the [examples folder](https://github.com/KonstantinLeyde/CosmoPyro/tree/main/examples) of this repository, which has its own README.

## Citation

If you use this software, please cite:

K. Leyde and E. Colangeli, *CosmoPyro: Gradients for Gravitational-Wave Cosmology*, [arXiv:2608.18281](https://arxiv.org/abs/2608.18281) (2026).

```bibtex
@article{Leyde:2026bat,
    author = "Leyde, Konstantin and Colangeli, Elena",
    title = "{CosmoPyro: Gradients for Gravitational-Wave Cosmology}",
    eprint = "2608.18281",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "8",
    year = "2026"
}
```

## Contact

If you have any questions, feedback, or would like to discuss new projects, please don't hesitate to reach out:

Email: Konstantin.Leyde@gmail.com