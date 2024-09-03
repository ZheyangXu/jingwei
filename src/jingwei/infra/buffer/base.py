from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Self, TypeVar

import numpy as np

from jingwei.transitions.base import Transition, TransitionBatch, TransitionMembers


class BaseBuffer(ABC):
    @abstractmethod
    def sample(self, batch_size: int) -> TransitionBatch:
        pass

    @property
    @abstractmethod
    def data(self) -> TransitionBatch:
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


class TransitionBuffer(BaseBuffer):
    def __init__(self, dtype: np.dtype = np.float32) -> None:  # type: ignore
        self.observation: np.ndarray = np.array([]) # shape: [num_episode, num_vec, observation.shape]
        self.action: np.ndarray = np.array([])
        self.reward: np.ndarray = np.array([])
        self.observation_next: np.ndarray = np.array([]) 
        self.terminated: np.ndarray = np.array([])
        self.truncated: np.ndarray = np.array([]) 
        self.done: np.ndarray = np.array([]) 
        self.dtype = dtype

    def __len__(self) -> int:
        if self.reward is None:
            return 0
        return len(self.reward)

    def __getitem__(self, index: int | str) -> Transition:
        if isinstance(index, str):
            if not TransitionMembers.contains(index):
                raise KeyError(f"{index} is not in keys {TransitionMembers.names()}")
            return self.__dict__[index]
        if not isinstance(index, int):
            raise TypeError(f"index shoud be str or int")
        if index >= self.__len__():
            raise IndexError("index out of range.")
        return Transition(
            self.observation[index],
            self.action[index],
            self.reward[index],
            self.observation_next[index],
            self.terminated[index],
            self.truncated[index],
        )

    def __iadd__(self, other: Self | Transition) -> Self:
        if isinstance(other, Transition):
            self._iadd_transition(other)
        elif isinstance(other, type(self)):
            self._iadd(other)
        return self

    def __add__(self, other: Self | Transition) -> Self:
        return deepcopy(self).__iadd__(other)

    def _iadd_transition(self, transition: Transition) -> None:
        for key, value in self.__dict__.items():
            if not TransitionMembers.contains(key):
                continue
            if value is None:
                self.__dict__[key] = np.array([getattr(transition, key)])
            else:
                self.__dict__[key] = np.append(self.__dict__[key], getattr(transition, key))

    def _iadd(self, other: Self) -> None:
        for key, value in self.__dict__.items():
            if not TransitionMembers.contains(key):
                continue

            if value is None:
                self.__dict__[key] = getattr(other, key)
            else:
                self.__dict__[key] = np.concatenate((self.__dict__[key], getattr(other, key)))

    def size(self) -> int:
        return self.__len__()

    def push(self, transition: Transition | Self) -> int:
        self.__iadd__(transition)
        return self.size()

    def as_type(self, dtype: np.dtype) -> None:
        self.dtype = dtype

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
