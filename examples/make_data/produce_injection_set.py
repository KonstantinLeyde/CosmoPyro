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

mmin, mmax = 2.5, 180
cfg = {
    "num_samples": 1_500_000,
    "seed": 123435,
    "snr_threshold": 12.0,
    "amplitude": 4e3,
    "cosmology": {"H0": 70.0, "Omega_m": 0.3},
    "redshift": {"gamma": 1.0, "kappa": 3.0, "zp": 2.0},
    "mass_model": {
        "mass_1_s": {
            "alpha": 2.1,
            "mmin": mmin,
            "mmax": mmax,
            "mu_g_low": 30.0,
            "sigma_g_low": 5.0,
            "lambda_g_low": 0.1,
            "delta_m": 1.0,
            "mu_g_high": 60.0,
            "sigma_g_high": 10.0,
            "lambda_g": 0.1,
        },
        "mass_ratio": {
            "beta_0": 1.5,
            "beta_1": 0.00,
            "sigma_mass_cutoff_mass_2": 0.5,
            "mass_ratio_running_zero_point": 10.0,
        },
    },
    "binning": {
        "m1_min": mmin,
        "m1_max": mmax,
        "m1_nbins": 10_000,
        "q_min": 0.03,
        "q_max": 1.0,
        "q_nbins": 4000,
        "nz_grid": 150_000,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--catalog_idx", type=int, default=0)
args = parser.parse_args()

num_catalogs = 200
catalog_idx = args.catalog_idx
data_dir = "../../data/delta_catalogs/"
catalog_names = []

for i in range(num_catalogs):
    cfg["seed"] += 1
    catalog_name = f"injections_{catalog_idx}_{i}.hdf5"
    catalog_names.append(catalog_name)

    print(f"Generating injections {i + 1}/{num_catalogs}: {catalog_name}")
    _ = generate_catalog(cfg, out_path=f"{data_dir}{catalog_name}")

# combine catalogs
all_data = []
for catalog_name in catalog_names:
    data = load_hdf5_to_namespace(f"{data_dir}{catalog_name}")
    all_data.append(data)

data_all = concatenate_namespaces(all_data)

save_namespace_to_hdf5(data_all, f"{data_dir}injections_{catalog_idx}.hdf5")

# delete individual catalogs after combining
for catalog_name in catalog_names:
    os.remove(f"{data_dir}{catalog_name}")
