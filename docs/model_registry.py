"""
Central registry of mass model and cosmology model configurations.

Used by:
  - The interactive kwargs builder in the docs (dumped to JSON at build time)
  - Can be imported in code for validation or auto-discovery

To add a new mass model, add an entry to MASS_MODELS below.
To add a new cosmology model, add an entry to COSMO_MODELS below.
"""

MASS_MODELS = {
    "power_law_peak": {
        "description": "Single peak",
        "dist_names": {
            "mass_1_s": "power_law_peak",
            "mass_ratio": "mass_ratio_running_power_law_in_log",
            "redshift": "MadauDickinson",
        },
        "mass_priors": {
            "mass_1_s": {
                "alpha": {"type": "Uniform", "min": 1.5, "max": 6.0, "value": 3.5},
                "mmin": {"type": "Uniform", "min": 2.0, "max": 10.0, "value": 5.0},
                "mmax": {"type": "Uniform", "min": 50.0, "max": 200.0, "value": 87.0},
                "mu_g": {"type": "Uniform", "min": 20.0, "max": 50.0, "value": 34.0},
                "sigma_g": {"type": "Uniform", "min": 1.0, "max": 10.0, "value": 4.0},
                "lambda_peak": {
                    "type": "Uniform",
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.04,
                },
                "delta_m": {"type": "Uniform", "min": 0.001, "max": 10.0, "value": 5.0},
            },
            "mass_ratio": {
                "beta_0": {"type": "Uniform", "min": -2.0, "max": 4.0, "value": 1.1},
                "beta_1": {"type": "Delta", "min": -0.03, "max": 0.03, "value": 0.0},
                "sigma_mass_cutoff_mass_2": {
                    "type": "Uniform",
                    "min": 0.1,
                    "max": 10.0,
                    "value": 1.0,
                },
                "mass_ratio_running_zero_point": {
                    "type": "Delta",
                    "min": 5.0,
                    "max": 50.0,
                    "value": 10.0,
                },
            },
        },
        "bins": {
            "mass_1_s": {"min": 1.0, "max": 150.0, "num": 400},
            "mass_ratio": {"min": 0.03, "max": 1.0, "num": 200},
            "redshift": {"min": 0.0, "max": 5.0, "num": 1000},
        },
    },
    "power_law_peak2": {
        "description": "Multi-peak",
        "dist_names": {
            "mass_1_s": "power_law_peak2",
            "mass_ratio": "mass_ratio_running_power_law_in_log",
            "redshift": "MadauDickinson",
        },
        "mass_priors": {
            "mass_1_s": {
                "alpha": {"type": "Uniform", "min": 1.5, "max": 6.0, "value": 3.5},
                "mmin": {"type": "Uniform", "min": 2.0, "max": 10.0, "value": 5.0},
                "mmax": {"type": "Uniform", "min": 50.0, "max": 200.0, "value": 87.0},
                "lambda_g": {"type": "Uniform", "min": 0.0, "max": 1.0, "value": 0.04},
                "lambda_g_low": {
                    "type": "Uniform",
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.5,
                },
                "delta_m": {"type": "Uniform", "min": 0.001, "max": 10.0, "value": 5.0},
                "mu_g_low": {"type": "Uniform", "min": 5.0, "max": 15.0, "value": 10.0},
                "sigma_g_low": {
                    "type": "Uniform",
                    "min": 0.4,
                    "max": 5.0,
                    "value": 2.0,
                },
                "mu_g_high": {
                    "type": "Uniform",
                    "min": 15.0,
                    "max": 100.0,
                    "value": 35.0,
                },
                "sigma_g_high": {
                    "type": "Uniform",
                    "min": 0.4,
                    "max": 10.0,
                    "value": 5.0,
                },
            },
            "mass_ratio": {
                "beta_0": {"type": "Uniform", "min": -2.0, "max": 4.0, "value": 1.1},
                "beta_1": {"type": "Delta", "min": -0.03, "max": 0.03, "value": 0.0},
                "mass_ratio_running_zero_point": {
                    "type": "Delta",
                    "min": 5.0,
                    "max": 50.0,
                    "value": 10.0,
                },
            },
        },
        "bins": {
            "mass_1_s": {"min": 1.0, "max": 150.0, "num": 400},
            "mass_ratio": {"min": 0.03, "max": 1.0, "num": 200},
            "redshift": {"min": 0.0, "max": 5.0, "num": 1000},
        },
    },
    "power_law_peak2_partial_windowed": {
        # Same parameters as power_law_peak2; only the windowing differs (the
        # smooth turn-on is applied to the power law alone, as in icarogw).
        # Defaults follow examples/configs/powerlaw_2peaks_gwtc5.yaml.
        "description": "Multi-peak (power law windowed only)",
        "dist_names": {
            "mass_1_s": "power_law_peak2_partial_windowed",
            "mass_ratio": "mass_ratio_running_power_law_in_log",
            "redshift": "MadauDickinson",
        },
        "mass_priors": {
            "mass_1_s": {
                "alpha": {"type": "Uniform", "min": 1.5, "max": 12.0, "value": 3.5},
                "mmin": {"type": "Uniform", "min": 2.0, "max": 10.0, "value": 5.0},
                "mmax": {"type": "Uniform", "min": 55.0, "max": 200.0, "value": 87.0},
                "lambda_g": {"type": "Uniform", "min": 0.0, "max": 1.0, "value": 0.04},
                "lambda_g_low": {
                    "type": "Uniform",
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.5,
                },
                "delta_m": {"type": "Uniform", "min": 0.001, "max": 10.0, "value": 5.0},
                "mu_g_low": {"type": "Uniform", "min": 5.0, "max": 15.0, "value": 10.0},
                "sigma_g_low": {
                    "type": "Uniform",
                    "min": 0.4,
                    "max": 5.0,
                    "value": 2.0,
                },
                "mu_g_high": {
                    "type": "Uniform",
                    "min": 15.0,
                    "max": 100.0,
                    "value": 35.0,
                },
                "sigma_g_high": {
                    "type": "Uniform",
                    "min": 0.4,
                    "max": 15.0,
                    "value": 5.0,
                },
            },
            "mass_ratio": {
                "beta_0": {"type": "Uniform", "min": -2.0, "max": 4.0, "value": 1.1},
                "beta_1": {"type": "Delta", "min": -0.03, "max": 0.03, "value": 0.0},
                "mass_ratio_running_zero_point": {
                    "type": "Delta",
                    "min": 5.0,
                    "max": 50.0,
                    "value": 10.0,
                },
            },
        },
        "bins": {
            "mass_1_s": {"min": 0.9, "max": 390.0, "num": 2000},
            "mass_ratio": {"min": 0.01, "max": 1.0, "num": 200},
            "redshift": {"min": 0.0, "max": 8.0, "num": 2000},
        },
    },
    "fourier_gp_1D": {
        "description": "1D GP + parametrized q",
        "noise_shape_bins": {"rows": "mass_1_s"},
        "dist_names": {
            "mass_1_s": "fourier_gp_1D",
            "mass_ratio": "mass_ratio_running_power_law_in_log",
            "redshift": "MadauDickinson",
        },
        "mass_priors": {
            "mass_1_s": {
                "gaussian_F_whitened_spatial [shape]": {
                    "type": "Normal_shape",
                    "rows": 200,
                    "cols": 1,
                },
                "mass_min": {"type": "Uniform", "min": 2, "max": 10, "value": 5.0},
                "mass_max": {"type": "Uniform", "min": 50, "max": 150, "value": 80.0},
                "sigma_low_fractional": {
                    "type": "Uniform",
                    "min": 0.01,
                    "max": 0.1,
                    "value": 0.05,
                },
                "sigma_high_fractional": {
                    "type": "Uniform",
                    "min": 0.01,
                    "max": 0.1,
                    "value": 0.05,
                },
                "power_spectrum_amplitude": {
                    "type": "Delta",
                    "min": 0.1,
                    "max": 20.0,
                    "value": 5.0,
                },
                "power_spectrum_cutoff": {
                    "type": "Delta",
                    "min": 1.0,
                    "max": 100.0,
                    "value": 5.0,
                },
            },
            "mass_ratio": {
                "beta_0": {"type": "Uniform", "min": -2.0, "max": 4.0, "value": 1.1},
                "beta_1": {"type": "Delta", "min": -0.03, "max": 0.03, "value": 0.0},
                "sigma_mass_cutoff_mass_2": {
                    "type": "Uniform",
                    "min": 0.1,
                    "max": 10.0,
                    "value": 1.0,
                },
                "mass_ratio_running_zero_point": {
                    "type": "Delta",
                    "min": 5.0,
                    "max": 50.0,
                    "value": 10.0,
                },
            },
        },
        "bins": {
            "mass_1_s": {"min": 2.0, "max": 120.0, "num": 200},
            "mass_ratio": {"min": 0.03, "max": 1.0, "num": 200},
            "redshift": {"min": 0.0, "max": 5.0, "num": 1000},
        },
    },
    "fourier_gp_2D_logMdelta": {
        "description": "2D GP (log M, \u2013log q)",
        "noise_shape_bins": {
            "rows": "log_mass_total_s",
            "cols": "minus_log_mass_ratio",
        },
        "dist_names": {
            "source_frame_masses": "fourier_gp_2D_logMdelta",
            "redshift": "MadauDickinson",
        },
        "mass_priors": {
            "source_frame_masses": {
                "gaussian_F_whitened_spatial [shape]": {
                    "type": "Normal_shape",
                    "rows": 120,
                    "cols": 120,
                },
                "mass_min": {"type": "Uniform", "min": 2, "max": 10, "value": 5.0},
                "mass_max": {"type": "Uniform", "min": 30, "max": 120, "value": 80.0},
                "sigma_low_fractional": {
                    "type": "Uniform",
                    "min": 0.01,
                    "max": 0.2,
                    "value": 0.05,
                },
                "sigma_high_fractional": {
                    "type": "Uniform",
                    "min": 0.01,
                    "max": 0.2,
                    "value": 0.05,
                },
                "power_spectrum_amplitude": {
                    "type": "Delta",
                    "min": 0.01,
                    "max": 50.0,
                    "value": 10.0,
                },
                "power_spectrum_cutoff": {
                    "type": "Delta",
                    "min": 1.0,
                    "max": 200.0,
                    "value": 80.0,
                },
                "power_spectrum_relative_scale_log_mass_total_s_to_minus_log_mass_ratio": {
                    "type": "Delta",
                    "min": 0.1,
                    "max": 10.0,
                    "value": 1.0,
                },
                "power_law_reference_mass_1_s": {
                    "type": "Delta",
                    "min": -5.0,
                    "max": 5.0,
                    "value": -2.0,
                },
                "power_law_reference_mass_ratio": {
                    "type": "Delta",
                    "min": -5.0,
                    "max": 5.0,
                    "value": 1.5,
                },
            },
        },
        "bins": {
            "log_mass_total_s": {"min": 1.5, "max": 6.0, "num": 120},
            "minus_log_mass_ratio": {"min": 0.0, "max": 4.0, "num": 120},
            "redshift": {"min": 0.0, "max": 5.0, "num": 1000},
        },
    },
    "fourier_gp_2D_m1sq": {
        "description": "2D GP (m\u2081, q)",
        "noise_shape_bins": {"rows": "mass_1_s", "cols": "mass_ratio"},
        "dist_names": {
            "source_frame_masses": "fourier_gp_2D_m1sq",
            "redshift": "MadauDickinson",
        },
        "mass_priors": {
            "source_frame_masses": {
                "gaussian_F_whitened_spatial [shape]": {
                    "type": "Normal_shape",
                    "rows": 120,
                    "cols": 120,
                },
                "mass_min": {"type": "Uniform", "min": 2, "max": 10, "value": 5.0},
                "mass_max": {"type": "Uniform", "min": 30, "max": 120, "value": 80.0},
                "sigma_low_fractional": {
                    "type": "Uniform",
                    "min": 0.01,
                    "max": 0.2,
                    "value": 0.05,
                },
                "sigma_high_fractional": {
                    "type": "Uniform",
                    "min": 0.01,
                    "max": 0.2,
                    "value": 0.05,
                },
                "power_spectrum_amplitude": {
                    "type": "Delta",
                    "min": 0.01,
                    "max": 50.0,
                    "value": 10.0,
                },
                "power_spectrum_cutoff": {
                    "type": "Delta",
                    "min": 1.0,
                    "max": 200.0,
                    "value": 80.0,
                },
                "power_spectrum_relative_scale_mass_1_s_to_mass_ratio": {
                    "type": "Delta",
                    "min": 0.1,
                    "max": 10.0,
                    "value": 1.0,
                },
                "power_law_reference_mass_1_s": {
                    "type": "Delta",
                    "min": -5.0,
                    "max": 5.0,
                    "value": -2.0,
                },
                "power_law_reference_mass_ratio": {
                    "type": "Delta",
                    "min": -5.0,
                    "max": 5.0,
                    "value": 1.5,
                },
            },
        },
        "bins": {
            "mass_1_s": {"min": 2.0, "max": 120.0, "num": 120},
            "mass_ratio": {"min": 0.02, "max": 1.0, "num": 120},
            "redshift": {"min": 0.0, "max": 5.0, "num": 1000},
        },
    },
}

COSMO_MODELS = {
    "FlatLambdaCDM": {
        "description": "Standard flat \u039bCDM",
        "cosmo_priors": {},
    },
    "FlatLambdaCDM_GW_distance_cosine": {
        "description": "Modified GW distance (cosine basis)",
        "cosmo_priors": {
            "modified_ratio": {
                "alphas": {"type": "Normal_shape", "rows": 15, "cols": 1},
                "phases": {
                    "type": "Uniform",
                    "min": 0.0,
                    "max": 6.2831853,
                    "value": 3.14,
                    "shape": [15],
                },
                "zmax_b": {"type": "Delta", "min": 0.5, "max": 5.0, "value": 1.0},
                "z_tr": {"type": "LogUniform", "min": 0.001, "max": 1.0, "value": 0.1},
            },
        },
    },
    "FlatLambdaCDM_GW_distance_gp_integrated": {
        "description": "Modified GW distance (GP ratio)",
        "cosmo_priors": {
            "modified_ratio": {
                "ratio_gaussian_whitened_field": {
                    "type": "Normal_shape",
                    "rows": 100,
                    "cols": 1,
                },
                "ratio_power_spectrum_amplitude": {
                    "type": "Delta",
                    "min": 0.0001,
                    "max": 1.0,
                    "value": 0.001,
                },
            },
        },
    },
    "FlatLambdaCDM_GW_distance_cM": {
        "description": "Modified GW distance (c_M)",
        "cosmo_priors": {
            "modified_ratio": {
                "cM": {"type": "Uniform", "min": -10.0, "max": 10.0, "value": 0.0},
            },
        },
    },
}
