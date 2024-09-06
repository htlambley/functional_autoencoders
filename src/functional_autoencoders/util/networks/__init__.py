import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any, Callable


Initializer = Callable[[jax.random.PRNGKey, Sequence[int], Any], Any]


class MLP(nn.Module):
    """
    Multilayer perceptron (MLP) neural network mapping between vectors.
    """

    features: Sequence[int]
    """
    Widths for each layer of the MLP. The final value is the width of the output layer.
    """

    act: Callable = lambda x: nn.gelu(x)
    """
    The activation function used between hidden layers (no activation is used on the output layer).
    Default is the GELU activation (Hendrycks and Gimpel, 2016); others are available through the
    [`flax.linen.activation`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/activation_functions.html) 
    module.
    """

    kernel_init: Initializer = None
    bias_init: Initializer = None

    use_bias: bool = True
    """
    Enables the use of a bias for each layer, including the output layer.
    """

    @nn.compact
    def __call__(self, x):
        kwargs = {}
        if self.kernel_init is not None:
            kwargs["kernel_init"] = self.kernel_init
        if self.bias_init is not None:
            kwargs["bias_init"] = self.bias_init

        for feat in self.features[:-1]:
            x = self.act(nn.Dense(feat, use_bias=self.use_bias, **kwargs)(x))
        x = nn.Dense(self.features[-1], use_bias=self.use_bias)(x)
        return x


class CNN(nn.Module):
    features: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    act: Callable = lambda x: nn.gelu(x)
    is_transpose: bool = False

    @nn.compact
    def __call__(self, x):
        n_layers = len(self.features)
        conv_fn = nn.Conv if not self.is_transpose else nn.ConvTranspose
        for i in range(n_layers - 1):
            x = self.act(
                conv_fn(
                    features=self.features[i],
                    kernel_size=(self.kernel_sizes[i], self.kernel_sizes[i]),
                    strides=(self.strides[i], self.strides[i]),
                )(x)
            )
        x = conv_fn(
            features=self.features[n_layers - 1],
            kernel_size=(
                self.kernel_sizes[n_layers - 1],
                self.kernel_sizes[n_layers - 1],
            ),
            strides=(self.strides[n_layers - 1], self.strides[n_layers - 1]),
        )(x)
        return x


class MultiheadAttentionBlock(nn.Module):
    n_heads: int
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2

    @nn.compact
    def __call__(self, X, Y):
        MH = MultiheadLinearAttentionLayer(n_heads=self.n_heads)(X, Y, Y)
        H = nn.LayerNorm()(X + MH)
        rFF = MLP([*[self.mlp_dim] * self.mlp_n_hidden_layers, H.shape[-1]])(H)
        X_out = nn.LayerNorm()(H + rFF)
        return X_out


class MultiheadLinearAttentionLayer(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, X_q, X_k, X_v):
        heads = []
        for _ in range(self.n_heads):
            h = AttentionLayer(dim_attn=X_v.shape[-1])(X_q, X_k, X_v)
            heads.append(h)

        heads_concat = jnp.concatenate(heads, axis=-1)

        X_out = nn.Dense(X_v.shape[-1], use_bias=False)(heads_concat)
        return X_out


class AttentionLayer(nn.Module):
    dim_attn: int

    @nn.compact
    def __call__(self, X_q, X_k, X_v):
        Q = nn.Dense(self.dim_attn, use_bias=False)(X_q)
        K = nn.Dense(self.dim_attn, use_bias=False)(X_k)
        V = nn.Dense(self.dim_attn, use_bias=False)(X_v)

        P = jnp.einsum("bkd,bnd->bkn", Q, K)
        P = P / jnp.sqrt(self.dim_attn)
        P = jax.nn.softmax(P, axis=-1)

        X_out = jnp.einsum("bkn,bnd->bkd", P, V)
        return X_out
