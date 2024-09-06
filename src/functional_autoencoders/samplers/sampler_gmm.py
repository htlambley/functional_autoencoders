import jax.numpy as jnp
from sklearn import mixture
from functional_autoencoders.samplers import SamplerBase


class SamplerGMM(SamplerBase):
    def __init__(self, autoencoder, state, n_components):
        super().__init__(autoencoder, state)
        self.gmm = None
        self.n_components = n_components

    def fit(self, train_dataloader):
        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=2000,
            verbose=0,
            tol=1e-3,
        )

        z_dataset = self._get_z_dataset(train_dataloader)
        gmm.fit(z_dataset)

        self.gmm = gmm

    def sample(self, x):
        z_samples, _ = self.gmm.sample(x.shape[0])
        u_samples = self.autoencoder.decode(self.state, z_samples, x, train=False)
        return u_samples

    def _get_z_dataset(self, train_dataloader):
        z_dataset = []
        for u, x, _, _ in train_dataloader:
            z = self.autoencoder.encode(self.state, u, x, train=False)
            z_dataset.append(z)
        z_dataset = jnp.concatenate(z_dataset, axis=0)
        return z_dataset
