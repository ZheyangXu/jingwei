from typing import Self

from httpx import delete
import numpy as np

from jingwei.infra.buffer.base import BaseBuffer
from jingwei.infra.typing import *
from jingwei.transitions.base import Transition, TransitionBatch, TransitionMembers


class ReplayBuffer(BaseBuffer):
    def __init__(self, capacity: int, dtype: np.dtype = np.float32) -> None:  # type: ignore
        super().__init__()
        self._capacity: int = capacity
        self.observation: np.ndarray = np.array([])
        self.action: np.ndarray = np.array([])
        self.reward: np.ndarray = np.array([])
        self.observation_next: np.ndarray = np.array([])
        self.terminated: np.ndarray = np.array([])
        self.truncated: np.ndarray = np.array([])
        self.done: np.ndarray = np.array([])
        self.dtype = dtype

    def __len__(self) -> int:
        return self.observation.shape[0]

    def __getitem__(self, index: int | slice | np.ndarray) -> Transition | TransitionBatch:
        if isinstance(index, int):
            return self.__get_transition(index)
        if isinstance(index, slice) or isinstance(index, np.ndarray):
            return self.__get_transition_batch(index)
        raise TypeError("Index should be int or slice.")

    def __get_transition(self, index: int) -> Transition:
        return Transition(
            self.observation[index],
            self.action[index],
            self.reward[index],
            self.observation_next[index],
            self.terminated[index],
            self.truncated[index],
        )

    def __get_transition_batch(self, index: slice | np.ndarray) -> TransitionBatch:
        return TransitionBatch(
            self.observation[index],
            self.action[index],
            self.reward[index],
            self.observation_next[index],
            self.terminated[index],
            self.truncated[index],
        )

    def capacity(self) -> int:
        return self._capacity

    def len(self) -> int:
        return self.__len__()

    def sample(self, batch_size: int) -> TransitionBatch:
        index = np.random.randint(0, self.len(), size=(batch_size,))
        return self.__get_transition_batch(index)

    def push(self, transition: Transition) -> int:
        if self.len() + 1 > self.capacity():
            self.delete(0)
        self.observation = self._concatenate(self.observation, transition.observation)
        if isinstance(transition.action, int):
            self.action = np.append(self.action, transition.action)
        else:
            self.action = self._concatenate(self.action, transition.action)
        self.reward = np.append(self.reward, transition.reward)
        self.observation_next = self._concatenate(
            self.observation_next, transition.observation_next
        )
        self.terminated = np.append(self.terminated, transition.terminated)
        self.truncated = np.append(self.truncated, transition.truncated)
        return self.len()
    
    def delete(self, index: int | slice) -> int:
        for key in TransitionMembers.names():
            setattr(self, key, np.delete(getattr(self, key), index, 0))
        return self.len()

    @staticmethod
    def _concatenate(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        if len(arr1) == 0:
            return np.expand_dims(arr2, 0)
        return np.concatenate((arr1, np.expand_dims(arr2, 0)), axis=0)

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
        return self.len()
