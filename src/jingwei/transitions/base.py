from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum, auto
from typing import List, NamedTuple, TypeVar

import numpy as np
import torch


class TransitionMembers(Enum):
    observation = auto()
    action = auto()
    reward = auto()
    observation_next = auto()
    terminated = auto()
    truncated = auto()
    done = auto()
    
    @staticmethod
    def names() -> List[str]:
        return TransitionMembers._member_names_
    
    @staticmethod
    def contains(key: str) -> bool:
        return key in TransitionMembers._member_names_


class Transition(NamedTuple):
    observation: np.ndarray
    action: np.ndarray | int
    reward: float
    observation_next: np.ndarray
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class TransitionBatch(NamedTuple):
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    observation_next: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def done(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)


class TensorTransitionBatch(NamedTuple):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    observation_next: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

    @property
    def done(self) -> torch.Tensor:
        return torch.logical_or(self.terminated, self.truncated)

    def to(self, device: torch.device) -> None:
        for key in TransitionMembers.names():
            getattr(self, key).to(device)
