import jax
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace

from cosmopyro.utils.utils import is_latent_sample_site


def test_factor_sites_are_not_latent_mass_matrix_sites():
    def model():
        numpyro.sample("theta", dist.Normal(0.0, 1.0))
        numpyro.factor("penalty_factor_relative_variance", -1.0)

    exec_trace = trace(seed(model, jax.random.key(0))).get_trace()

    latent_sites = [
        name for name, site in exec_trace.items() if is_latent_sample_site(site)
    ]

    assert latent_sites == ["theta"]
    assert "penalty_factor_relative_variance" not in latent_sites
