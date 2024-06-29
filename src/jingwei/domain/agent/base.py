from abc import ABC, abstractmethod

from jingwei.infra.typing import *
from jingwei.transitions.base import TransitionBatch


class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, observation: ObservationType) -> ActionType:
        pass

    @abstractmethod
    def estimate_return(self, transitions: TransitionBatch) -> ValueType:
        pass

    @abstractmethod
    def update_step(self, transitions: TransitionBatch) -> None:
        pass

    @abstractmethod
    def compute_loss(self, transitions: TransitionBatch) -> TensorType:
        pass
