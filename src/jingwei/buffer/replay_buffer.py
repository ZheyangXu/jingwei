from typing import Self, Tuple

import numpy as np

from jingwei.domain.buffer import Buffer
from jingwei.infra.typing import *
from jingwei.transitions.base import Transition, TransitionBatch


class ReplayBuffer(Buffer):
    def __init__(self, buffer_size: int, observation_shape: Tuple[int, ...], action_shape: Tuple[int, ...] | int, num_envs: int = 1, dtype: np.dtype = np.float32) -> None:  # type: ignore
        super().__init__(buffer_size, observation_shape, action_shape, num_envs, dtype)

    def get(self, batch_size: int) -> TransitionBatch:
        index = np.random.randint(0, self._pos, size=(batch_size,))
        return self._get_transition_batch(index)

    @property
    def data(self) -> TransitionBatch:
        return TransitionBatch(
            self.observation,
            self.action,
            self.reward,
            self.observation_next,
            self.terminated,
            self.truncated,
        )
