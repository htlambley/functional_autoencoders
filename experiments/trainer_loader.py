import jax
import numpy as np
from functional_autoencoders.domains import grid
from functional_autoencoders.domains import off_grid
from functional_autoencoders.encoders.cnn_encoder import CNNEncoder
from functional_autoencoders.encoders.pooling_encoder import PoolingEncoder
from experiments.custom_encoders import DiracEncoder
from experiments.custom_decoders import DiracDecoder
from functional_autoencoders.decoders.cnn_decoder import CNNDecoder
from functional_autoencoders.decoders.nonlinear_decoder import NonlinearDecoder
from functional_autoencoders.positional_encodings import (
    RandomFourierEncoding,
    IdentityEncoding,
)
from functional_autoencoders.util.networks.pooling import DeepSetPooling
from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.losses.vano import get_loss_vano_fn
from functional_autoencoders.losses.fvae_sde import get_loss_fvae_sde_fn
from functional_autoencoders.losses.fae import get_loss_fae_fn
from functional_autoencoders.train.autoencoder_trainer import AutoencoderTrainer
from functional_autoencoders.train.metrics import MSEMetric


def get_trainer(key, config, train_dataloader, test_dataloader):
    key, subkey = jax.random.split(key)
    autoencoder = get_autoencoder(subkey, config)

    domain = get_domain(config, train_dataloader)
    loss_fn = get_loss_fn(config, autoencoder, domain)
    metrics = get_metrics(config, autoencoder, domain)

    trainer = AutoencoderTrainer(
        autoencoder=autoencoder,
        loss_fn=loss_fn,
        metrics=metrics,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    return trainer


def get_autoencoder(key, config):
    key, subkey = jax.random.split(key)
    positional_encoding = get_positional_encoding(subkey, config)

    encoder = get_encoder(config, positional_encoding)
    decoder = get_decoder(config, positional_encoding)

    autoencoder = Autoencoder(
        encoder=encoder,
        decoder=decoder,
    )

    return autoencoder


def get_positional_encoding(key, config):
    component_config = config["positional_encoding"]

    if component_config["is_used"]:
        key, subkey = jax.random.split(key)
        b_mat = jax.random.normal(subkey, (component_config["dim"] // 2, 2))
        positional_encoding = RandomFourierEncoding(B=b_mat)
    else:
        positional_encoding = IdentityEncoding()

    return positional_encoding


def get_encoder(config, positional_encoding):
    component_config = config["encoder"]
    component_type = component_config["type"]
    hyperparams = component_config["options"][component_type]

    if component_type == "pooling":
        pooling_fn = DeepSetPooling(
            mlp_dim=hyperparams["mlp_dim"],
            mlp_n_hidden_layers=hyperparams["mlp_n_hidden_layers"],
        )
        encoder = PoolingEncoder(
            latent_dim=component_config["latent_dim"],
            is_variational=component_config["is_variational"],
            pooling_fn=pooling_fn,
            positional_encoding=positional_encoding,
        )
    elif component_type == "dirac":
        encoder = DiracEncoder(
            latent_dim=component_config["latent_dim"],
            is_variational=component_config["is_variational"],
            features=hyperparams["features"],
        )
    elif component_type == "cnn":
        encoder = CNNEncoder(
            latent_dim=component_config["latent_dim"],
            is_variational=component_config["is_variational"],
            cnn_features=hyperparams["cnn_features"],
            kernel_sizes=hyperparams["kernel_sizes"],
            strides=hyperparams["strides"],
            mlp_features=hyperparams["mlp_features"],
        )
    else:
        raise ValueError(f"Unknown encoder type: {component_type}")

    return encoder


def get_decoder(config, positional_encoding):
    component_config = config["decoder"]
    component_type = component_config["type"]
    hyperparams = component_config["options"][component_type]

    if component_type == "nonlinear":
        decoder = NonlinearDecoder(
            out_dim=hyperparams["out_dim"],
            features=hyperparams["features"],
            positional_encoding=positional_encoding,
        )
    elif component_type == "dirac":
        decoder = DiracDecoder(
            fixed_centre=False,
            features=hyperparams["features"],
            min_std=lambda dx: (1 / np.sqrt(2 * np.pi)) * dx,
        )
    elif component_type == "cnn":
        decoder = CNNDecoder(
            trans_cnn_features=hyperparams["trans_cnn_features"],
            kernel_sizes=hyperparams["kernel_sizes"],
            strides=hyperparams["strides"],
            mlp_features=hyperparams["mlp_features"],
            final_cnn_features=hyperparams["final_cnn_features"],
            final_kernel_sizes=hyperparams["final_kernel_sizes"],
            final_strides=hyperparams["final_strides"],
            c_in=hyperparams["c_in"],
            grid_pts_in=hyperparams["grid_pts_in"],
        )
    else:
        raise ValueError(f"Unknown decoder type: {component_type}")

    return decoder


def get_domain(config, train_dataloader):
    component_config = config["domain"]
    component_type = component_config["type"]
    hyperparams = component_config["options"][component_type]

    if component_type == "grid_zero_boundary_conditions":
        domain = grid.ZeroBoundaryConditions(
            s=hyperparams["s"],
        )
    elif component_type == "off_grid_randomly_sampled_euclidean":
        domain = off_grid.RandomlySampledEuclidean(
            s=hyperparams["s"],
        )
    elif component_type == "off_grid_sde":
        domain = off_grid.SDE(
            epsilon=config["data"]["epsilon"], x0=train_dataloader.dataset.x0[0]
        )
    else:
        raise ValueError(f"Unknown domain type: {component_type}")

    return domain


def get_loss_fn(config, autoencoder, domain):
    component_config = config["loss"]
    component_type = component_config["type"]
    hyperparams = component_config["options"][component_type]

    if component_type == "fae":
        loss_fn = get_loss_fae_fn(
            autoencoder=autoencoder,
            domain=domain,
            beta=hyperparams["beta"],
            subtract_data_norm=hyperparams["subtract_data_norm"],
        )
    elif component_type == "vano":
        loss_fn = get_loss_vano_fn(
            autoencoder=autoencoder,
            rescale_by_norm=hyperparams["rescale_by_norm"],
            normalised_inner_prod=hyperparams["normalised_inner_prod"],
            beta=hyperparams["beta"],
            n_monte_carlo_samples=hyperparams["n_monte_carlo_samples"],
        )
    elif component_type == "fvae_sde":
        loss_fn = get_loss_fvae_sde_fn(
            autoencoder=autoencoder,
            domain=domain,
            beta=hyperparams["beta"],
            theta=hyperparams["theta"],
            zero_penalty=hyperparams["zero_penalty"],
            n_monte_carlo_samples=hyperparams["n_monte_carlo_samples"],
        )
    else:
        raise ValueError(f"Unknown loss type: {component_type}")

    return loss_fn


def get_metrics(config, autoencoder, domain):
    metrics = [MSEMetric(autoencoder, domain=domain)]
    return metrics
