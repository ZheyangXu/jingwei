from abc import ABC, abstractmethod

from jingwei.infra.mtype import MType
from jingwei.infra.typing import *
from jingwei.transitions.base import TensorTransitionBatch


class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, observation: ObservationType) -> ActionType:
        pass

    @abstractmethod
    def estimate_return(self, transitions: TensorTransitionBatch) -> ValueType:
        pass

    @abstractmethod
    def update_step(self, transitions: TensorTransitionBatch) -> None:
        pass

    @abstractmethod
    def compute_actor_loss(self, transitions: TensorTransitionBatch) -> TensorType:
        pass
    
    @property
    @abstractmethod
    def mtype(self) -> MType:
        pass
