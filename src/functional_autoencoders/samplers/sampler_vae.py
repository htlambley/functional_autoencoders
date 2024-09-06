import jax
from functional_autoencoders.samplers import SamplerBase


class SamplerVAE(SamplerBase):
    def __init__(self, autoencoder, state):
        super().__init__(autoencoder, state)

    def sample(self, x, key):
        latent_dim = self.autoencoder.get_latent_dim()
        latents = jax.random.normal(key, [x.shape[0], latent_dim])
        decoded = self.autoencoder.decode(self.state, latents, x, train=False)
        return decoded
