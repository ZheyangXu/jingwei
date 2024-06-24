from abc import ABC, abstractmethod

from jingwei.infra.typing import *
from jingwei.transitions.base import TransitionBatch


class BaseActor(ABC):
    @abstractmethod
    def get_action(self, observation: ObservationType) -> ActionType:
        pass

    @abstractmethod
    def get_probs(self, transitions: TransitionBatch) -> TensorType:
        pass

    @abstractmethod
    def get_log_probs(self, transitions: TransitionBatch) -> TensorType:
        pass

    @abstractmethod
    def update_step(self, loss: LossType) -> None:
        pass
