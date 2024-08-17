from abc import ABC, abstractmethod

from jingwei.domain.distributions.base import Distribution
from jingwei.infra.typing import *
from jingwei.transitions.base import TransitionBatch


class BaseActor(ABC):
    @abstractmethod
    def get_action(self, observation: ObservationTensor, deterministic: bool = False) -> ActionType:
        pass

    @abstractmethod
    def get_probs(self, observation: ObservationTensor) -> TensorType:
        pass

    @abstractmethod
    def get_probs(self, observation: ObservationTensor) -> TensorType:
        pass

    @abstractmethod
    def get_log_probs(self, observation: ObservationTensor) -> TensorType:
        pass

    @abstractmethod
    def to(self, device: DeviceType = None) -> DeviceType:
        pass
