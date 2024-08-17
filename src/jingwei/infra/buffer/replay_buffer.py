import numpy as np

from jingwei.infra.buffer.base import BaseBuffer
from jingwei.infra.typing import *
from jingwei.transitions.base import Transition, TransitionBatch, TransitionMembers


class ReplayBuffer(BaseBuffer):
    def __init__(self, capacity: int, dtype: np.dtype = np.float32) -> None:
        super().__init__()
        self.capacity = capacity
        self.observation: np.ndarray = None
        self.action: np.ndarray = None
        self.reward: np.ndarray = None
        self.terminated: np.ndarray = None
        self.truncated: np.ndarray = None
        self.done: np.ndarray = None
        self.dtype = dtype

    def __len__(self) -> int:
        if self.observation is None:
            return 0
        return self.observation.shape[0]

    def __getitem__(self, index: int | slice) -> Transition | TransitionBatch:
        if isinstance(index, int):
            return Transition(
                self.observation[index],
                self.action[index],
                self.reward[index],
                self.observation_next[index],
                self.terminated[index],
                self.truncated[index],
            )
        if isinstance(index, slice):
            return TransitionBatch(
                self.observation[index],
                self.action[index],
                self.reward[index],
                self.observation_next[index],
                self.terminated[index],
                self.truncated[index],
            )
        raise TypeError("Index should be int or slice.")

    def capacity(self) -> int:
        return self.capacity

    def len(self) -> int:
        return self.__len__()
