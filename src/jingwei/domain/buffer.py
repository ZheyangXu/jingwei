from abc import ABC, abstractmethod
from copy import deepcopy
from textwrap import indent
from typing import Self, Tuple

import numpy as np

from jingwei.infra.typing import *
from jingwei.transitions.base import Transition, TransitionBatch, TransitionMembers


class BaseBuffer(ABC):
    @abstractmethod
    def get(self, batch_size: int) -> TransitionBatch:
        pass

    @abstractmethod
    def push(self, transition: Transition) -> int:
        pass

    @abstractmethod
    def capacity(self) -> int:
        pass

    @abstractmethod
    def len(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> int:
        pass


class Buffer(BaseBuffer):
    def __init__(self, capacity: int, observation_shape: Tuple[int, ...], action_shape: Tuple[int, ...] | int | np.int64, num_envs: int = 1, dtype: np.dtype = None) -> None:  # type: ignore
        self._capacity: int = capacity
        self.observation_shape: Tuple[int, ...] = (capacity, *observation_shape)
        self.action_shape = (
            (capacity, action_shape)
            if isinstance(action_shape, (int | np.int64))
            else (capacity, *action_shape)
        )
        self.shape: Tuple[int, ...] = (capacity,)
        self.dtype = dtype
        self._pos = 0

        self.observation: ObservationType = np.zeros(self.observation_shape, dtype=self.dtype)
        self.action: ActionType = np.zeros(self.action_shape, dtype=self.dtype)
        self.reward: RewardType = np.zeros(self.shape, dtype=self.dtype)
        self.observation_next: ObservationType = np.zeros(self.observation_shape, dtype=self.dtype)
        self.terminated: DoneType = np.zeros(self.shape, dtype=self.dtype)
        self.truncated: DoneType = np.zeros(self.shape, dtype=self.dtype)
        self.done: DoneType = np.zeros(self.shape, dtype=self.dtype)

    def __len__(self) -> int:
        return self._pos

    def __getitem__(self, index: int | slice | np.ndarray) -> Transition | TransitionBatch:
        if isinstance(index, int):
            if index > self._pos:
                raise IndexError("index out of range.")
            return self.__get_transition(index)
        if isinstance(index, (slice, np.ndarray)):
            return self.__get_transition_batch(index)
        raise TypeError("Index should be int or slice.")

    def _get_transition(self, index: int) -> Transition:
        return Transition(
            self.observation[index],
            self.action[index],
            self.reward[index],
            self.observation_next[index],
            self.terminated[index],
            self.truncated[index],
        )

    def _get_transition_batch(self, index: slice | np.ndarray) -> TransitionBatch:
        return TransitionBatch(
            self.observation[index],
            self.action[index],
            self.reward[index],
            self.observation_next[index],
            self.terminated[index],
            self.truncated[index],
        )

    def push(self, transition: Transition | Self) -> int:
        self.observation[self._pos] = transition.observation
        self.action[self._pos] = transition.action
        self.reward[self._pos] = transition.reward
        self.observation_next[self._pos] = transition.observation_next
        self.terminated[self._pos] = transition.terminated
        self.truncated[self._pos] = transition.truncated
        self.done[self._pos] = transition.done
        self._pos += 1
        if self._pos >= self._capacity:
            self._pos = 0
        return self._pos

    def capacity(self) -> int:
        return self.capacity()

    def len(self) -> int:
        return self._pos

    def clear(self) -> int:
        self.observation: np.ndarray = np.zeros(self.observation_shape, dtype=self.dtype)
        self.action: np.ndarray = np.zeros(self.action_shape, dtype=self.dtype)
        self.reward: np.ndarray = np.zeros(self.shape, dtype=self.dtype)
        self.observation_next: np.ndarray = np.zeros(self.observation_shape, dtype=self.dtype)
        self.terminated: np.ndarray = np.zeros(self.shape, dtype=self.dtype)
        self.truncated: np.ndarray = np.zeros(self.shape, dtype=self.dtype)
        self.done: np.ndarray = np.zeros(self.shape, dtype=self.dtype)
        self._pos = 0
        return self.len()

    def get(self, batch_size: int) -> TransitionBatch:
        raise NotImplementedError
