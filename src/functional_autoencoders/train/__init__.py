import jax
from flax.core import FrozenDict
from flax.training import train_state


class TrainNanError(Exception):
    pass


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict
    key: jax.Array
