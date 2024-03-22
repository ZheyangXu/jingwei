# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod
from typing import Optional

import torch

from jingwei.infra.typing import ActionType, RewardType, StateType


class ReplayBuffer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._is_action_continuous: bool = False
        self._has_cost_available: bool = False

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @device.setter
    @abstractmethod
    def device(self, device: torch.device | str) -> None:
        pass

    @abstractmethod
    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: RewardType,
        next_state: StateType,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> object:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def is_action_continuous(self) -> bool:
        return self._is_action_continuous

    @is_action_continuous.setter
    def is_action_continuous(self, value: bool) -> None:
        self._is_action_continuous = value
