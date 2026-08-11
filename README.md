<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/KonstantinLeyde/CosmoPyro/refs/heads/main/images/CosmoPyro_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/KonstantinLeyde/CosmoPyro/refs/heads/main/images/CosmoPyro_light.png">
    <img alt="CosmoPyro logo" src="https://raw.githubusercontent.com/KonstantinLeyde/CosmoPyroV1/refs/heads/main/images/CosmoPyro_light.png" width="280">
  </picture>
</div>

# **CosmoPyro**

CosmoPyro is a Bayesian framework for reconstructing the gravitational-wave population of compact binary coalescences using Hamiltonian Monte Carlo (HMC), and more specifically the No-U-Turn sampler implemented in [NumPyro](https://num.pyro.ai/en/stable/).

## Installation Instructions

### Installation from PyPI

If you want to install CosmoPyro as quick as possible, you can use 

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
cd cosmopyro
uv pip install -e .
```

### Making the Environment GPU Compatible
To run on GPUs, you'll need the GPU version of `jax` matching your cluster's CUDA version:

```bash
pip install --upgrade --force-reinstall "jax[cuda12]"
```

## Running

For examples, see either this dedicated [git repository](https://github.com/KonstantinLeyde/CosmoPyro-Demo) or navigate to the [examples folder](examples), and follow the readme there. notebooks in the examples folder.

## Citation

If you use this software, please cite:

[TO WRITE](https://arxiv.org/pdf/??)

```bibtex
@article{
    TOCOLLATE
}
```

## Contact

If you have any questions, feedback, or would like to discuss this project, please don't hesitate to reach out:

Email: Konstantin.Leyde@gmail.com