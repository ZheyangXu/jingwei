from typing import Self, Tuple

import numpy as np

from jingwei.domain.buffer import Buffer
from jingwei.infra.typing import *
from jingwei.transitions.base import *


class RolloutBuffer(Buffer):
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...] | int,
        num_envs: int = 1,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(capacity, observation_shape, action_shape, num_envs, dtype)
        self._last_pos = 0

    def get(self, batch_size: int) -> TransitionBatch:
        index = range(self._last_pos, min(self._last_pos + batch_size, self._pos))
        self._last_pos += batch_size
        if self._last_pos >= self._pos:
            self._last_pos = 0
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

    def clear(self) -> int:
        super().clear()
        self._last_pos = 0
        return self.len()
