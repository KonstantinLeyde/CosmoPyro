import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

os.environ["JAX_PLATFORMS"] = "cpu"

from cosmopyro.data.data_utils import (
    load_hdf5_to_namespace,
    save_namespace_to_hdf5,
)
from cosmopyro.data.generate_catalogs import (
    concatenate_namespaces,
    generate_catalog,
)

cfg = {
    "num_samples": 100_000,
    "seed": 42371234,
    "snr_threshold": 12.0,
    "amplitude": 4e3,
    "cosmology": {"H0": 67.0, "Omega_m": 0.319},
    "redshift": {"gamma": 2.7, "kappa": 3.0, "zp": 2.0},
    "mass_model": {
        "mass_1_s": {
            "alpha": 2.61, "mmin": 5.24, "mmax": 69.44,
            "lambda_g": 0.44, "delta_m": 3.05,
            "lambda_g_low": 0.91,
            "mu_g_low": 9.49, "sigma_g_low": 0.45,
            "mu_g_high": 31.76, "sigma_g_high": 1.4,
        },
        "mass_ratio": {
            "beta_0": 0.85, "beta_1": 0.0,
            "sigma_mass_cutoff_mass_2": 3.05,
            "mass_ratio_running_zero_point": 10.0,
        },
    },
    "binning": {
        "m1_min": 4.0, "m1_max": 100.0, "m1_nbins": 10_000,
        "q_min": 0.03, "q_max": 1.0, "q_nbins": 4000, "nz_grid": 40_000,
    },
    "cosmology_name": "FlatLambdaCDM",
}

data_dir = "../../data/delta_catalogs/"

parser = argparse.ArgumentParser()
parser.add_argument("--catalog_idx", type=int, default=0)
args = parser.parse_args()

cfg['seed'] += args.catalog_idx * 1000

catalog_idx = args.catalog_idx
num_catalogs = 10
catalog_names = []

for i in range(num_catalogs):
    cfg['seed'] += 1
    catalog_name = f"catalog_{catalog_idx}_{i}.hdf5"
    catalog_names.append(catalog_name)

    print(f"Generating catalog {i + 1}/{num_catalogs}: {catalog_name}")
    _ = generate_catalog(cfg, out_path=f"{data_dir}{catalog_name}")

# produce reference catalog with no cut
cfg_ref = cfg.copy()
cfg_ref["snr_threshold"] = -1.0
cfg_ref["num_samples"] = 500_000
catalog_name_ref = f"catalog_{catalog_idx}_reference.hdf5"
_ = generate_catalog(cfg_ref, out_path=f"{data_dir}{catalog_name_ref}")

# combine catalogs (except reference)
all_data = []
for catalog_name in catalog_names:
    data = load_hdf5_to_namespace(f"{data_dir}{catalog_name}")
    all_data.append(data)

data_all = concatenate_namespaces(all_data)

save_namespace_to_hdf5(data_all, f'{data_dir}catalog_{catalog_idx}_combined.hdf5')

# delete individual catalogs after combining
for catalog_name in catalog_names:
    os.remove(f"{data_dir}{catalog_name}")
