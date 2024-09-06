import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from functional_autoencoders.util.networks import MLP, Initializer
from functional_autoencoders.domains import Domain


class LNOLayer(nn.Module):
    n_rank: int
    R_init: Initializer = nn.initializers.glorot_normal()
    act = nn.gelu
    mlp_hidden_features: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, u, x):
        x = x / jnp.max(jnp.abs(x))
        Wu = nn.Dense(u.shape[-1], use_bias=False)(u)

        f = MLP([*self.mlp_hidden_features, 2 * self.n_rank * u.shape[-1]])(x)
        f = f.reshape(*x.shape[:-1], 2 * self.n_rank, u.shape[-1])
        phi, psi = jnp.split(f, 2, axis=-2)
        l2_inner_prods = (psi * jnp.expand_dims(u, axis=-2)).mean(axis=-3)
        l2_inner_prods = jnp.expand_dims(l2_inner_prods, axis=-3)
        Ku = jnp.sum(l2_inner_prods * phi, axis=-2)

        return self.act(Wu + Ku)


class LNO(nn.Module):
    domain: Domain
    n_ranks: Sequence[int]
    lifting_features: Sequence[int]
    projection_features: Sequence[int]
    act = None
    R_init: Initializer = None
    mlp_init: Initializer = None
    lno_mlp_hidden_features: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, u, x):
        fno_kwargs = {}
        mlp_kwargs = {}

        if self.mlp_init is not None:
            mlp_kwargs["initializer"] = self.mlp_init

        if self.act is not None:
            fno_kwargs["act"] = self.act
            mlp_kwargs["act"] = self.act

        if self.R_init is not None:
            fno_kwargs["R_init"] = self.R_init

        u = MLP(self.lifting_features, **mlp_kwargs)(u)
        for n_rank in self.n_ranks:
            u = LNOLayer(
                n_rank=n_rank,
                mlp_hidden_features=self.lno_mlp_hidden_features,
                **fno_kwargs,
            )(u, x)
        u = MLP(self.projection_features, **mlp_kwargs)(u)
        return u
