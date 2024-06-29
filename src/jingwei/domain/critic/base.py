from abc import ABC, abstractmethod

from jingwei.infra.typing import *
from jingwei.infra.typing import LossType
from jingwei.transitions.base import TensorTransitionBatch, TransitionBatch


class BaseCritic(ABC):
    @abstractmethod
    def estimate_return(self, transitions: TransitionBatch) -> ValueType:
        pass

    @abstractmethod
    def update_step(self, loss: LossType) -> None:
        pass


class CriticTrait(BaseCritic):
    def __init__(self, model: ModelType, optimizer: OptimizerType, device: DeviceType) -> None:
        super().__init__()
        self.model = model
        self.optimzer = optimizer
        self.device = device
        self.model.to(self.device)

    def estimate_return(self, transitions: TransitionBatch) -> TensorType:
        return self.model(transitions.observation)

    def update_step(self, loss: LossType) -> None:
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()

    def to(self, device: DeviceType = None) -> DeviceType:
        if device is None:
            device = self.device
        self.model.to(device)
        return device
