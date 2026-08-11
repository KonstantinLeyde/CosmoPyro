"""
Generate a GW event catalog using a galaxy catalog as host locations.

Usage:
    python produce_catalog_from_galaxy_catalog.py --galaxy_catalog path/to/galaxies.hdf5
"""

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORMS"] = "cpu"

from cosmopyro.data.data_utils import load_hdf5_to_namespace
from cosmopyro.data.generate_catalogs import generate_catalog_from_galaxy_catalog

cfg = {
    "num_samples": 1_000_000,
    "seed": 42,
    "snr_threshold": 12.0,
    "amplitude": 4e3,
    "cosmology": {"H0": 70.0, "Omega_m": 0.3},
    "mass_model": {
        "mass_1_s": {
            "alpha": 3.1,
            "mmin": 5.0,
            "mmax": 70.0,
            "lambda_g": 0.1,
            "delta_m": 3.0,
            "lambda_g_low": 0.7,
            "mu_g_low": 25.0,
            "sigma_g_low": 3.0,
            "mu_g_high": 35.0,
            "sigma_g_high": 7.0,
        },
        "mass_ratio": {
            "beta_0": 1.2,
            "beta_1": 0.02,
            "sigma_mass_cutoff_mass_2": 0.5,
            "mass_ratio_running_zero_point": 10.0,
        },
    },
    "binning": {
        "m1_min": 5.0,
        "m1_max": 100.0,
        "m1_nbins": 1650,
        "q_min": 0.03,
        "q_max": 1.0,
        "q_nbins": 1600,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--galaxy_catalog", type=str, required=True)
parser.add_argument(
    "--out_path",
    type=str,
    default="../../data/delta_catalogs/catalog_galaxy_hosts.hdf5",
)
args = parser.parse_args()

galaxy_catalog = load_hdf5_to_namespace(args.galaxy_catalog)
catalog = generate_catalog_from_galaxy_catalog(
    cfg, galaxy_catalog, out_path=args.out_path
)

print(
    f"Generated {catalog.redshift.shape[0]} events (from {int(cfg['num_samples'])} draws)"
)
print(f"Saved to {args.out_path}")
