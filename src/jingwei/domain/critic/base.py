from abc import ABC, abstractmethod

from jingwei.infra.typing import *
from jingwei.transitions.base import TensorTransitionBatch, TransitionBatch


class BaseCritic(ABC):
    @abstractmethod
    def estimate_return(self, transitions: TransitionBatch) -> ValueType:
        pass

    @abstractmethod
    def update_step(self, loss: LossType) -> None:
        pass


