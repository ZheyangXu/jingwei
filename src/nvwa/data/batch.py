from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from nvwa.data.transition import Transition


@dataclass(frozen=True)
class Batch(object):
    observation: NDArray
    action: NDArray
    reward: NDArray
    observation_next: NDArray
    terminated: NDArray
    truncated: NDArray

    def observation_dtype(self) -> np.dtype:
        return self.observation.dtype

    def action_dtype(self) -> np.dtype:
        return self.action.dtype

    def __len__(self) -> int:
        return len(self.reward)

    def __add__(self, other: Self | Transition) -> Self:
        if isinstance(other, Transition):
            self.observation = np.concatenate(
                (self.observation, other.observation[None, ...]), axis=0
            )
            self.action = np.concatenate((self.action, other.action[None, ...]), axis=0)
            self.reward = np.concatenate((self.reward, other.reward[None, ...]), axis=0)
            self.observation_next = np.concatenate(
                (self.observation_next, other.observation_next[None, ...]), axis=0
            )
            self.terminated = np.concatenate((self.terminated, other.terminated[None, ...]), axis=0)
            self.truncated = np.concatenate((self.truncated, other.truncated[None, ...]), axis=0)
        elif isinstance(other, Self):
            self.observation = np.concatenate((self.observation, other.observation), axis=0)
            self.action = np.concatenate((self.action, other.action), axis=0)
            self.reward = np.concatenate((self.reward, other.reward), axis=0)
            self.observation_next = np.concatenate(
                (self.observation_next, other.observation_next), axis=0
            )
            self.terminated = np.concatenate((self.terminated, other.terminated), axis=0)
            self.truncated = np.concatenate((self.truncated, other.truncated), axis=0)
        else:
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")

    def __iadd__(self, other: Self | Transition) -> Self:
        return self.__add__(other)


@dataclass(frozen=True)
class RolloutBatch(Batch):
    log_prob: NDArray
    values: NDArray
    prob: NDArray
    advantages: NDArray

    def __add__(self, other: Self | Transition) -> Self:
        super().__add__(other)
        if isinstance(other, Transition):
            self.log_prob = np.concatenate((self.log_prob, other.log_prob[None, ...]), axis=0)
            self.values = np.concatenate((self.values, other.values[None, ...]), axis=0)
            self.prob = np.concatenate((self.prob, other.prob[None, ...]), axis=0)
            self.advantages = np.concatenate((self.advantages, other.advantages[None, ...]), axis=0)
        elif isinstance(other, Self):
            self.log_prob = np.concatenate((self.log_prob, other.log_prob), axis=0)
            self.values = np.concatenate((self.values, other.values), axis=0)
            self.prob = np.concatenate((self.prob, other.prob), axis=0)
            self.advantages = np.concatenate((self.advantages, other.advantages), axis=0)
        else:
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")

    def __iadd__(self, other: Self | Transition) -> Self:
        return self.__add__(other)
