# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod

from jingwei.infra.typing import *


class BaseCritic(ABC):
    @abstractmethod
    def estimate(self, state: StateType) -> ValueType:
        pass

    @abstractmethod
    def update_fn(self, loss: LossType) -> None:
        pass
