import jax
import flax
from flax.training import train_state


class TrainNanError(Exception):
    pass


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict
    key: jax.Array
