import os
import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

os.environ["JAX_PLATFORMS"] = "cpu"

from cosmopyro.data.data_utils import (
    load_hdf5_to_namespace,
    save_namespace_to_hdf5,
)
from cosmopyro.data.generate_simple_pe import compute_posterior_samples

parser = argparse.ArgumentParser()
parser.add_argument("--catalog_idx", type=int)
args = parser.parse_args()

data_dir = "../../data/delta_catalogs/"

catalog_idx = args.catalog_idx
path = f"{data_dir}catalog_{catalog_idx}_combined.hdf5"

data = load_hdf5_to_namespace(path)

concentration = {
    'chirp_mass_d': (10.0, 1000.0),          # relative uncertainty ~ 1/sqrt(concentration+1), modulo SNR contribution to posterior
    'mass_ratio': (3.0, 7.0),
    'luminosity_distance': (1.0, 5.0),
}

samples = compute_posterior_samples(
    data_all=data,
    num_events=1000,
    num_posterior_samples=15000,
    uncertainties="gamma",
    seed=192,
    concentration=concentration,
)

save_namespace_to_hdf5(samples, f"{data_dir}posterior_samples_{catalog_idx}.hdf5")
