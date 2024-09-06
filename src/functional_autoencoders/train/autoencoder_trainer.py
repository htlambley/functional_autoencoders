import sys
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from typing import Sequence, Literal, Union
from functional_autoencoders.util import get_n_params
from functional_autoencoders.train import (
    TrainState,
    TrainNanError,
)
from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.train.metrics import Metric


class AutoencoderTrainer:
    autoencoder: Autoencoder
    metrics: Sequence[Metric]

    def __init__(
        self,
        autoencoder: Autoencoder,
        loss_fn,
        metrics: Sequence[Metric],
        train_dataloader,
        test_dataloader,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.metrics_history = {}
        self.training_loss_history = []

    def _get_verbosity_level(self, verbose) -> Literal["full", "metrics", "none"]:
        if isinstance(verbose, bool):
            return "full" if verbose else "none"
        else:
            return verbose

    def fit(
        self,
        key,
        lr,
        lr_decay_step,
        lr_decay_factor,
        max_step,
        eval_interval=10,
        verbose: Union[bool, Literal["full", "metrics", "none"]] = False,
    ):
        """
        Fits the `AutoencoderTrainer` to the training data provided by `train_dataloader` in the constructor,
        using the validation data from `test_dataloader` in the constructor to compute evaluation metrics.

        :param key: JAX pseudorandom number generator key (`jax.random.PRNGKey`)
        :param lr: learning rate
        :param lr_decay_step: along with `lr_decay_factor`, parameters for the [`optax.exponential_decay`](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.exponential_decay) learning-rate scheduler
        :param lr_decay_factor: see `lr_decay_step`
        :param max_step: training will finish when an epoch is complete and the total number of steps exceeds `max_step`
        :param eval_interval: number of epochs to train before a validation step
        :param verbose: verbosity of the `AutoencoderTrainer`. `"full"` shows a progress bar per epoch (for use in, e.g., interactive sessions). `"metrics"` prints only evaluation metrics every `eval_interval` epochs. `"none"` prints nothing.
        """
        self._init_history()
        verbose = self._get_verbosity_level(verbose)

        key, subkey = jax.random.split(key)
        state = self._get_init_state(subkey, lr, lr_decay_step, lr_decay_factor)

        if verbose != "none":
            print(f"Parameter count: {get_n_params(state.params)}")

        train_step_fn = self._get_train_step_fn()

        epoch = 0
        step = 0
        while True:
            key, subkey = jax.random.split(key)
            state, step = self._train_one_epoch(
                subkey, state, step, train_step_fn, epoch, verbose
            )

            if epoch % eval_interval == 0:
                key, subkey = jax.random.split(key)
                self._evaluate(subkey, state)
                self._print_metrics(epoch, verbose)

            if step >= max_step:
                return {
                    "state": state,
                    "training_loss_history": self.training_loss_history,
                    "metrics_history": self.metrics_history,
                }

            epoch += 1

    def _train_one_epoch(self, key, state, step, train_step_fn, epoch, verbose):
        epoch_loss = 0.0
        for i, batch in enumerate(
            pbar := tqdm(
                self.train_dataloader,
                disable=(verbose != "full"),
                desc=f"epoch {epoch}",
            )
        ):
            key, subkey = jax.random.split(key)
            loss_value, state = train_step_fn(
                subkey,
                state,
                batch,
            )

            epoch_loss += loss_value
            step += 1

            if verbose == "full":
                pbar.set_description(f"epoch {epoch} (loss {loss_value:.3E})")
            if jnp.any(jnp.isnan(epoch_loss)):
                raise TrainNanError()

        epoch_loss /= i + 1
        self.training_loss_history.append(epoch_loss)

        return state, step

    def _evaluate(self, key, state):
        for metric in self.metrics:
            key, subkey = jax.random.split(key)
            self.metrics_history[metric.name].append(
                metric(state, subkey, self.test_dataloader)
            )

    def _print_metrics(self, epoch, verbose):
        if verbose != "none":
            metric_string = " | ".join(
                [
                    f"{metric_name}: {self.metrics_history[metric_name][-1]:.3E}"
                    for metric_name in self.metrics_history
                ]
            )
            print(f"epoch {epoch:6} || {metric_string}")
            sys.stdout.flush()

    def _get_optimizer(self, lr, lr_decay_step, lr_decay_factor):
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=lr_decay_step,
            decay_rate=lr_decay_factor,
        )
        optimizer = optax.adam(learning_rate=schedule)
        return optimizer

    def _get_init_variables(self, key):
        key, subkey = jax.random.split(key)
        init_u, init_x, _, _ = next(iter(self.train_dataloader))
        variables = self.autoencoder.init(subkey, init_u, init_x, init_x)
        return variables

    def _get_init_state(self, key, lr, lr_decay_step, lr_decay_factor):
        optimizer = self._get_optimizer(lr, lr_decay_step, lr_decay_factor)

        key, subkey = jax.random.split(key)
        init_variables = self._get_init_variables(subkey)

        key, subkey = jax.random.split(key)
        state = TrainState.create(
            apply_fn=self.autoencoder.apply,
            params=init_variables["params"],
            tx=optimizer,
            batch_stats=(
                init_variables["batch_stats"]
                if "batch_stats" in init_variables
                else None
            ),
            key=subkey,
        )
        return state

    def _init_history(self):
        self.metrics_history = {metric.name: [] for metric in self.metrics}
        self.training_loss_history = []

    def _get_train_step_fn(self):
        @jax.jit
        def step_func(k, state, batch):
            u_dec, x_dec, u_enc, x_enc = batch
            grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
            (loss_value, batch_stats), grads = grad_fn(
                state.params,
                key=k,
                batch_stats=state.batch_stats,
                u_enc=u_enc,
                x_enc=x_enc,
                u_dec=u_dec,
                x_dec=x_dec,
            )
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=batch_stats)
            return loss_value, state

        return step_func
