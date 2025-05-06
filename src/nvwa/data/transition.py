from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Transition(object):
    observation: NDArray
    action: NDArray | np.int64
    reward: float
    observation_next: NDArray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class RolloutTransition(Transition):
    log_prob: float
    values: float
    prob: float
