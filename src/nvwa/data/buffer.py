from abc import ABC, abstractmethod
from typing import Generator, List

import gymnasium as gym
import numpy as np
import torch

from nvwa.data.batch import Batch
from nvwa.data.transition import RolloutTransition, Transition
from nvwa.infra.functional import get_action_dimension, get_observation_shape
from nvwa.infra.wrapper import DataWrapper


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.observation_shape = get_observation_shape(observation_space)
        self.action_dimension = get_action_dimension(action_space)
        self.device = device
        self.dtype = dtype
        self.wrapper = DataWrapper(observation_space, action_space, dtype, device)
        self._init_buffer()
        self.__keys = [
            "observation",
            "action",
            "reward",
            "observation_next",
            "terminated",
            "truncated",
        ]

    def _init_buffer(self) -> int:
        self.pos = 0
        self.full = False
        self.observation = np.zeros(
            (self.buffer_size, *self.observation_shape), dtype=self.observation_space.dtype
        )
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action = np.zeros((self.buffer_size, 1), dtype=self.action_space.dtype)
        else:
            self.action = np.zeros(
                (self.buffer_size, self.action_dimension), dtype=self.action_space.dtype
            )
        self.reward = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.observation_next = np.zeros_like(self.observation)
        self.terminated = np.zeros((self.buffer_size, 1), dtype=np.bool_)
        self.truncated = np.zeros((self.buffer_size, 1), dtype=np.bool_)
        self._episode_end_position = []
        return self.size()

    def __getitem__(self, key: str) -> torch.Tensor | np.ndarray:
        if key not in self.__keys:
            raise ValueError(f"Key {key} does not exist in the buffer.")
        return getattr(self, key)

    def __setitem__(self, key: str, value: torch.Tensor | np.ndarray) -> None:
        if key not in self.__keys:
            self.__keys.append(key)
        setattr(self, key, value)

    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def is_full(self) -> bool:
        return self.full

    def put(self, transition: Transition, *args, **kwargs) -> int:
        self.observation[self.pos] = transition.observation
        self.action[self.pos] = transition.action
        self.observation_next[self.pos] = transition.observation_next
        self.reward[self.pos] = transition.reward
        self.terminated[self.pos] = transition.terminated
        self.truncated[self.pos] = transition.truncated
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
        return self.size()

    def add(self, batch: Batch) -> int:
        end_position = self.pos + len(batch)
        for key in batch.keys():
            batch_data = batch.get(key)
            if batch_data is None:
                continue
            if not hasattr(self, key):
                setattr(self, key, np.zeros((self.buffer_size, 1), dtype=batch.get(key).dtype))
            data = getattr(self, key)
            if isinstance(data, List):
                data.append(batch_data)
            else:
                data[self.pos : end_position] = batch_data
            setattr(self, key, data)
        if batch._episode_end_position is not None:
            self._episode_end_position.extend(batch._episode_end_position)
        self.pos = end_position
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
        return self.size()

    def reset(self) -> int:
        self._init_buffer()
        return self.pos

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
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(buffer_size, observation_space, action_space, device, dtype)

    def _get_batch(self, batch_indices: List[int]) -> Batch:
        return Batch(
            observation=self.wrapper.wrap_observation(self.observation[batch_indices]),
            action=self.wrapper.wrap_action(self.action[batch_indices]),
            reward=self.wrapper.to_tensor(self.reward[batch_indices]),
            observation_next=self.wrapper.wrap_observation(self.observation_next[batch_indices]),
            terminated=self.wrapper.to_tensor(self.terminated[batch_indices]),
            truncated=self.wrapper.to_tensor(self.truncated[batch_indices]),
        )


class RolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(RolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, dtype
        )
        self._init_buffer()

    def _init_buffer(self) -> int:
        super()._init_buffer()
        self.value = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.old_log_prob = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.advantage = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.episode_index = np.zeros((self.buffer_size, 1), dtype=np.int32)
        self.old_log_prob = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dist = []
        self._episode_end_position = []
        return self.size()

    def put(self, rollout_transition: RolloutTransition) -> int:
        super().put(rollout_transition)
        self.value[self.pos] = rollout_transition.values
        self.log_prob[self.pos] = rollout_transition.log_prob
        return self.size()

    def _get_batch(self, batch_indices: List[int], episode: int = 0) -> Batch:
        return Batch(
            observation=self.wrapper.wrap_observation(self.observation[batch_indices]),
            action=self.wrapper.wrap_action(self.action[batch_indices]),
            reward=self.wrapper.to_tensor(self.reward[batch_indices]),
            observation_next=self.wrapper.wrap_observation(self.observation_next[batch_indices]),
            terminated=self.wrapper.to_tensor(self.terminated[batch_indices]),
            truncated=self.wrapper.to_tensor(self.truncated[batch_indices]),
            advantage=self.wrapper.to_tensor(self.advantage[batch_indices]),
            returns=self.wrapper.to_tensor(self.returns[batch_indices]),
            old_log_prob=self.wrapper.to_tensor(self.old_log_prob[batch_indices]),
            dist=self.dist[episode] if episode < len(self.dist) else None,
        )

    def get_batch(self, batch_size: int) -> Generator[Batch, None, None]:
        indices = np.random.permutation(self.size())
        start_index = 0
        while start_index < self.size():
            yield self._get_batch(indices[start_index : start_index + batch_size])
            start_index += batch_size

    def get_episode_batch(self) -> Generator[Batch, None, None]:
        for idx, end_position in enumerate(self._episode_end_position):
            batch_indices = np.arange(end_position)
            yield self._get_batch(batch_indices, idx)
