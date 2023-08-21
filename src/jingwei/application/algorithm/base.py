# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod
from typing import List

from jingwei.infra.typing import *
from jingwei.domain.experience.base import Transition


class BaseAlgorithm(ABC):
    @abstractmethod
    def take_action(self, state: StateType) -> ActionType:
        pass

    @abstractmethod
    def estimate_return(self, transitions: List[Transition]) -> ValueType:
        pass

    @abstractmethod
    def update_step(self, transitions: List[Transition]) -> None:
        pass
