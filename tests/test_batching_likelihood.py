import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

# Enable x64 as requested
jax.config.update("jax_enable_x64", True)

# Note: In a real scenario, you would import these from your actual package.
# Assuming the user has these available in the environment as per the previous snippet.
from cosmopyro.models.models import calculate_safe_log_prob_batched, get_log_prob


class MockCosmology:
    """Simulates the cosmology model methods required by get_log_prob."""

    def get_redshift_from_luminosity_distance_gw(
        self, luminosity_distance_gw, params_modified_gravity=None
    ):
        return luminosity_distance_gw / 1000.0

    def get_dluminosity_distance_gw_over_dz_from_redshift(
        self, redshift, params_modified_gravity=None
    ):
        return jnp.ones_like(redshift) * 100.0


class MockDistribution:
    """Simulates a conditional interpolated distribution."""

    def __init__(self, x_names, y_names):
        self.x_names = x_names
        self.y_names = y_names

    def log_prob(self, x_vals, y_vals):
        # Return a simple deterministic calculation based on inputs to verify data flow
        val = 0.0
        for v in x_vals.values():
            val += jnp.log(jnp.abs(v) + 1.0)
        for v in y_vals.values():
            val += jnp.log(jnp.abs(v) + 1.0)
        return val


# ==========================================
# 3. UNIT TEST
# ==========================================


class TestBatching(unittest.TestCase):
    def setUp(self):
        """Common setup for both tests."""
        # Mock Params
        self.params = {
            "cosmology": {"H0": 70.0},
        }

        # Mock Models
        self.cosmo_model = MockCosmology()

        # Create distributions
        dist_m1 = MockDistribution(x_names=["mass_1_s"], y_names=[])
        dist_q = MockDistribution(x_names=["mass_ratio"], y_names=[])
        dist_z = MockDistribution(x_names=["redshift"], y_names=["healpix_idx"])

        self.distributions = {0: dist_m1, 1: dist_q, 2: dist_z}

        self.rng = np.random.default_rng(42)

    def _generate_samples(self, shape):
        """Helper to generate random samples of specific shape."""
        return SimpleNamespace(
            luminosity_distance=jnp.array(self.rng.uniform(100, 5000, shape)),
            mass_1_d=jnp.array(self.rng.uniform(5, 100, shape)),
            mass_ratio=jnp.array(self.rng.uniform(0.1, 1.0, shape)),
            healpix_idx=jnp.array(self.rng.integers(0, 100, shape)),
            prior_masses_d_dL=jnp.array(self.rng.uniform(0.01, 0.1, shape)),
        )

    def test_calculate_safe_log_prob_batched_consistency(self):
        print("\n--- Running 2D (Standard) Batching Test ---")

        # 1. Setup Dimensions (2D)
        num_events = 100
        num_posterior_samples = 1000
        shape = (num_events, num_posterior_samples)

        samples = self._generate_samples(shape)

        # 2. Run Unbatched
        print("Running Unbatched Calculation (2D)...")
        log_prob_unbatched = get_log_prob(
            samples, self.cosmo_model, self.distributions, self.params
        )

        # 3. Run Batched
        # Total size is 100,000. Batch size 30,000 leaves remainder.
        batch_size = 30000
        print(f"Running Batched Calculation (Batch size: {batch_size})...")

        log_prob_batched = calculate_safe_log_prob_batched(
            samples,
            self.cosmo_model,
            self.distributions,
            self.params,
            batch_size=batch_size,
        )

        # 4. Verification
        self.assertEqual(log_prob_batched.shape, shape, "Output shape mismatch")

        is_close = jnp.allclose(
            log_prob_unbatched, log_prob_batched, atol=1e-12, rtol=1e-12
        )
        diff = jnp.max(jnp.abs(log_prob_unbatched - log_prob_batched))

        print(f"Max difference: {diff}")
        self.assertTrue(is_close, f"Batched result mismatch. Max diff: {diff}")

    def test_calculate_safe_log_prob_batched_flattened(self):
        print("\n--- Running 1D (Flattened) Batching Test ---")

        # 1. Setup Dimensions (1D)
        # Same total elements as above, but flattened into one dimension
        total_size = 100 * 1000
        shape = (total_size,)

        samples = self._generate_samples(shape)

        # 2. Run Unbatched
        print("Running Unbatched Calculation (1D)...")
        log_prob_unbatched = get_log_prob(
            samples, self.cosmo_model, self.distributions, self.params
        )

        # 3. Run Batched
        # Same batch logic, but applied to a 1D array
        batch_size = 30000
        print(f"Running Batched Calculation (Batch size: {batch_size})...")

        log_prob_batched = calculate_safe_log_prob_batched(
            samples,
            self.cosmo_model,
            self.distributions,
            self.params,
            batch_size=batch_size,
        )

        # 4. Verification
        # Ensure the output is also flattened
        self.assertEqual(log_prob_batched.shape, shape, "Output shape should be 1D")

        is_close = jnp.allclose(
            log_prob_unbatched, log_prob_batched, atol=1e-12, rtol=1e-12
        )
        diff = jnp.max(jnp.abs(log_prob_unbatched - log_prob_batched))

        print(f"Max difference: {diff}")
        self.assertTrue(
            is_close, f"Batched result mismatch on flattened input. Max diff: {diff}"
        )

    def test_gradients_match_unbatched(self):
        """Batched (scan + remainder + checkpoint) gradients equal unbatched ones."""
        samples = self._generate_samples((100_000,))
        distributions = self.distributions
        params = self.params

        class ParamCosmology:
            def __init__(self, h):
                self.h = h

            def get_redshift_from_luminosity_distance_gw(self, luminosity_distance_gw):
                return luminosity_distance_gw / (1000.0 * self.h)

            def get_dluminosity_distance_gw_over_dz_from_redshift(self, redshift):
                return 1000.0 * self.h * jnp.ones_like(redshift)

        def make_loss(batch_size):
            def loss(h):
                lp = calculate_safe_log_prob_batched(
                    samples,
                    ParamCosmology(h),
                    distributions,
                    params,
                    batch_size=batch_size,
                )
                return jnp.sum(lp)

            return loss

        value_unbatched, grad_unbatched = jax.value_and_grad(make_loss(10**9))(0.7)
        # 30_000 does not divide 100_000, so this exercises the remainder path too
        value_batched, grad_batched = jax.value_and_grad(make_loss(30_000))(0.7)

        self.assertTrue(jnp.isfinite(grad_batched))
        self.assertTrue(
            jnp.allclose(value_unbatched, value_batched, rtol=1e-12),
            f"Batched value mismatch: {value_unbatched} vs {value_batched}",
        )
        self.assertTrue(
            jnp.allclose(grad_unbatched, grad_batched, rtol=1e-12),
            f"Batched gradient mismatch: {grad_unbatched} vs {grad_batched}",
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
