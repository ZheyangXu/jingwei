# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod

from jingwei.infra.typing import ActionType, StateType, LossType


class BaseActor(ABC):
    @abstractmethod
    def take_action(self, state: StateType) -> ActionType:
        pass
    
    @abstractmethod
    def update_fn(self, loss: LossType) -> None:
        pass
