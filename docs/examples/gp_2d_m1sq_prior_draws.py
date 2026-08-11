import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib.pyplot as plt

from cosmopyro.distributions.mass_distributions_gaussian_process import (
    construct_prob_nn_whitened_field_2D_m1sq,
)
from cosmopyro.utils.jax_utils import compute_centers_and_delta_from_array

# Set up binning
n_m1, n_q = 120, 120
m1_edges = jnp.linspace(2.0, 120.0, n_m1 + 1)
q_edges = jnp.linspace(0.02, 1.0, n_q + 1)

m1_centers, m1_deltas = compute_centers_and_delta_from_array(m1_edges)
q_centers, q_deltas = compute_centers_and_delta_from_array(q_edges)

analysis = SimpleNamespace(
    binning=dict(
        boundaries=dict(mass_1_s=m1_edges, mass_ratio=q_edges),
        centers=dict(mass_1_s=m1_centers, mass_ratio=q_centers),
        deltas=dict(mass_1_s=m1_deltas, mass_ratio=q_deltas),
    )
)

# Draw 6 realizations
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for i, ax in enumerate(axes.flat):
    key = jax.random.PRNGKey(i)
    noise = jax.random.normal(key, shape=(n_m1, n_q))

    params = dict(
        source_frame_masses=dict(
            gaussian_F_whitened_spatial=noise,
            mass_min=5.0,
            mass_max=80.0,
            sigma_low_fractional=0.05,
            sigma_high_fractional=0.05,
            power_spectrum_amplitude=0.045,
            power_spectrum_cutoff=20.0,
            power_spectrum_relative_scale_mass_1_s_to_mass_ratio=1.0,
        )
    )

    prob_m1q = construct_prob_nn_whitened_field_2D_m1sq(analysis, params)

    ax.pcolormesh(
        m1_centers,
        q_centers,
        prob_m1q.T,
        shading="auto",
        cmap="viridis",
    )
    ax.set_xlabel(r"$m_{1,s}$")
    ax.set_ylabel(r"$q$")
    ax.set_title(f"Prior draw {i + 1}")

plt.tight_layout()
