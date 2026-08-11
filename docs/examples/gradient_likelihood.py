"""Gradient of the full CosmoPyro likelihood w.r.t. two parameters."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import Predictive

from cosmopyro.models import models
from cosmopyro.utils import analyses

# --- Load analysis ---
SETTINGS_PATH = "../../results/YOUR_JOB_ID/kwargs_analysis.yaml"  # <-- CHANGE THIS

result = analyses.Result(
    models.model_evaluate_p_theta,
    settings_path=SETTINGS_PATH,
)
data, skymap = result.load_data()
result.set_data_kwargs(data=data, skymap=skymap)

# Draw a single prior sample as baseline
rng_key = jax.random.PRNGKey(42)
prior_sample = Predictive(models.model_evaluate_p_theta, num_samples=1)(
    rng_key, **result.data_kwargs
)
base_params = {k: v.squeeze(0) for k, v in prior_sample.items()}


# --- Define log-likelihood as a function of (h, gamma) ---
def log_likelihood(h, gamma):
    params = {**base_params, "h": h, "gamma": gamma}
    model_trace = trace(
        substitute(seed(models.model_evaluate_p_theta, rng_key), params)
    ).get_trace(**result.data_kwargs)

    ll = sum(
        model_trace[s]["fn"].log_prob(model_trace[s]["value"]).sum()
        for s in ["log_prob_E", "log_prob_selection"]
    )
    return ll


grad_fn = jax.grad(log_likelihood, argnums=(0, 1))

# --- Evaluate on a grid ---
h_grid = jnp.linspace(0.5, 0.9, 8)
gamma_grid = jnp.linspace(0.5, 4.5, 8)
H, G = jnp.meshgrid(h_grid, gamma_grid)

ll_vmap = jax.vmap(jax.vmap(log_likelihood, in_axes=(0, 0)), in_axes=(0, 0))
grad_vmap = jax.vmap(jax.vmap(grad_fn, in_axes=(0, 0)), in_axes=(0, 0))

Z = ll_vmap(H, G)
Gh, Gg = grad_vmap(H, G)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.contourf(H, G, Z, levels=20, cmap="viridis")
ax1.set_xlabel("h")
ax1.set_ylabel("gamma")
ax1.set_title("log L(h, gamma)")

ax2.quiver(H, G, Gh, Gg, color="white")
ax2.contourf(H, G, Z, levels=20, cmap="viridis", alpha=0.4)
ax2.set_xlabel("h")
ax2.set_ylabel("gamma")
ax2.set_title("Gradient of log L")

plt.tight_layout()
plt.savefig("gradient_likelihood.png", dpi=150)
