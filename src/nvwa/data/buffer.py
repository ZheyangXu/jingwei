from abc import ABC, abstractmethod
from typing import List

import gymnasium as gym
import numpy as np

from nvwa.data.batch import Batch, RolloutBatch
from nvwa.data.transition import RolloutTransition, Transition
from nvwa.infra.functional import get_action_dimension, get_observation_shape


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ) -> None:
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.observation_shape = get_observation_shape(observation_space)
        self.action_dimension = get_action_dimension(action_space)
        self.pos = 0
        self.full = False

    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def put(self, *args, **kwargs) -> int:
        return self.size()

    def extend(self, *args, **kwargs) -> int:
        for data in zip(*args):
            self.put(*data, **kwargs)
        return self.size()

    def sample(self, batch_size: int) -> Batch:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_indices = np.random.choice(upper_bound, batch_size)
        batch = self._get_batch(batch_indices)
        return batch

    @abstractmethod
    def _get_batch(self, batch_indices: np.ndarray) -> Batch: ...


class ReplayBuffer(BaseBuffer):
    def __init__(
        self, buffer_size: int, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
    ) -> None:
        super().__init__(buffer_size, observation_space, action_space)
        self._init_buffer()

    def _init_buffer(self) -> int:
        self.observation = np.zeros(
            (self.buffer_size, *self.observation_shape), dtype=self.observation_space.dtype
        )
        self.action = np.zeros(
            (self.buffer_size, self.action_dimension), dtype=self.action_space.dtype
        )
        self.reward = np.zeros((self.buffer_size,), dtype=np.float32)
        self.observation_next = np.zeros_like(self.observation)
        self.terminated = np.zeros((self.buffer_size,), dtype=np.bool)
        self.truncated = np.zeros((self.buffer_size,), dtype=np.bool)
        return self.pos

    def put(self, transition: Transition) -> int:
        self.observation[self.pos] = transition.observation
        self.action[self.pos] = transition.action
        self.reward[self.pos] = transition.reward
        self.observation_next[self.pos] = transition.observation_next
        self.terminated[self.pos] = transition.terminated
        self.truncated[self.pos] = transition.truncated
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
        return self.size()

    def _get_batch(self, batch_indices: List[int]) -> Batch:
        return Batch(
            observation=self.observation[batch_indices],
            action=self.action[batch_indices],
            reward=self.reward[batch_indices],
            observation_next=self.observation_next[batch_indices],
            terminated=self.terminated[batch_indices],
            truncated=self.truncated[batch_indices],
        )


class RolloutBuffer(BaseBuffer):
    def __init__(
        self, buffer_size: int, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
    ) -> None:
        super().__init__(buffer_size, observation_space, action_space)
        self._init_buffer()

    def _init_buffer(self) -> int:
        self.observation = np.zeros(
            (self.buffer_size, *self.observation_shape), dtype=self.observation_space.dtype
        )
        self.action = np.zeros(
            (self.buffer_size, self.action_dimension), dtype=self.action_space.dtype
        )
        self.reward = np.zeros((self.buffer_size,), dtype=np.float32)
        self.observation_next = np.zeros_like(self.observation)
        self.terminated = np.zeros((self.buffer_size,), dtype=np.bool)
        self.truncated = np.zeros((self.buffer_size,), dtype=np.bool)
        self.value = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_prob = np.zeros((self.buffer_size,), dtype=np.float32)
        self.prob = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        return self.pos

    def put(self, rollout_transition: RolloutTransition) -> int:
        self.observation[self.pos] = rollout_transition.observation
        self.action[self.pos] = rollout_transition.action
        self.reward[self.pos] = rollout_transition.reward
        self.observation_next[self.pos] = rollout_transition.observation_next
        self.terminated[self.pos] = rollout_transition.terminated
        self.truncated[self.pos] = rollout_transition.truncated
        self.value[self.pos] = rollout_transition.values
        self.log_prob[self.pos] = rollout_transition.log_prob
        self.prob[self.pos] = rollout_transition.prob
        self.advantages[self.pos] = rollout_transition.advantages
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
        return self.size()

    def _get_batch(self, batch_indices: List[int]) -> RolloutBatch:
        return RolloutBatch(
            observation=self.observation[batch_indices],
            action=self.action[batch_indices],
            reward=self.reward[batch_indices],
            observation_next=self.observation_next[batch_indices],
            terminated=self.terminated[batch_indices],
            truncated=self.truncated[batch_indices],
            log_prob=self.log_prob[batch_indices],
            values=self.value[batch_indices],
            prob=self.prob[batch_indices],
            advantages=self.advantages[batch_indices],
        )
