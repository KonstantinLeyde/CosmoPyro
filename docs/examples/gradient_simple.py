"""Simple 2D gradient example with JAX: function + gradient arrows."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# A 2D function: correlated Gaussian
def log_prob(x, y):
    return -0.5 * (x**2 + y**2 - 0.8 * x * y)


# JAX gives us the gradient for free
grad_fn = jax.grad(log_prob, argnums=(0, 1))

# Evaluate on a grid
x = jnp.linspace(-3, 3, 50)
y = jnp.linspace(-3, 3, 50)
X, Y = jnp.meshgrid(x, y)
Z = jax.vmap(jax.vmap(log_prob, in_axes=(0, 0)), in_axes=(0, 0))(X, Y)

# Gradient arrows on a coarser grid
xq = jnp.linspace(-2.5, 2.5, 10)
yq = jnp.linspace(-2.5, 2.5, 10)
Xq, Yq = jnp.meshgrid(xq, yq)
grad_vmap = jax.vmap(jax.vmap(grad_fn, in_axes=(0, 0)), in_axes=(0, 0))
Gx, Gy = grad_vmap(Xq, Yq)

fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(X, Y, Z, levels=20, cmap="viridis")
ax.quiver(Xq, Yq, Gx, Gy, color="white", alpha=0.8)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("log p(x, y) with gradient arrows")
plt.tight_layout()
# plt.savefig('gradient_simple.png', dpi=150)
