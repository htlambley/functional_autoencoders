import jax.numpy as jnp
import flax.linen as nn
from functional_autoencoders.util.networks import MLP, MultiheadLinearAttentionLayer


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


class MultiheadAttentionPooling(nn.Module):
    n_heads: int = 2
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2

    @nn.compact
    def __call__(self, u, x):
        indices = jnp.arange(1, dtype=jnp.int32)
        z = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)
        s = nn.Embed(1, z.shape[-1])(indices)[None, :]
        s = jnp.repeat(s, z.shape[0], axis=0)

        z = MultiheadLinearAttentionLayer(n_heads=self.n_heads)(s, z, z)
        z = z[:, 0, :]
        return z


class DeepSetPooling(nn.Module):
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2

    @nn.compact
    def __call__(self, u, x):
        z = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)
        z = z.mean(axis=1)
        return z
