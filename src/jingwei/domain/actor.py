from abc import ABC, abstractmethod

from jingwei.infra.typing import *


class BaseActor(ABC):
    @abstractmethod
    def get_action(self, observation: ObservationTensor, deterministic: bool = False) -> ActionType:
        pass

    @abstractmethod
    def get_probs(self, observation: ObservationTensor) -> TensorType:
        pass

    @abstractmethod
    def get_log_probs(self, observation: ObservationTensor) -> TensorType:
        pass

    @abstractmethod
    def update_step(self, loss: TensorType) -> None:
        pass

    @abstractmethod
    def to(self, device: DeviceType = "") -> DeviceType:
        pass
