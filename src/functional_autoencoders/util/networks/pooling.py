import jax.numpy as jnp
import flax.linen as nn
from functional_autoencoders.util.networks import MLP


class MLPKernelPooling(nn.Module):
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2

    @nn.compact
    def __call__(self, u, x):
        u_dim = u.shape[-1]
        hidden_features = [self.mlp_dim] * self.mlp_n_hidden_layers
        mlp_features = [*hidden_features, self.mlp_dim * u_dim]

        kernel_eval_shape = [*x.shape[:-1], self.mlp_dim, u_dim]
        kernel_evals = MLP(mlp_features)(x).reshape(kernel_eval_shape)

        z = jnp.einsum("...xy,...y->...x", kernel_evals, u)
        z = z.mean(axis=range(1, z.ndim - 1))
        return z
