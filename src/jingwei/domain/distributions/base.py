from abc import ABC, abstractmethod
from typing import TypeVar

from jingwei.infra.typing import *


DistributionType = TypeVar("DistributionType", bound="Distribution")


class Distribution(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.distribution = None

    @abstractmethod
    def prob_distribution(self, **kwargs) -> DistributionType:
        pass

    @abstractmethod
    def log_prob(self, action: ActionType) -> TensorType:
        pass

    @abstractmethod
    def entropy(self) -> TensorType:
        pass

    @abstractmethod
    def sample(self) -> ActionType:
        pass

    @abstractmethod
    def mode(self) -> ActionType:
        pass

    def get_action(self, deterministic: bool = False) -> ActionType:
        if deterministic:
            return self.mode()
        return self.sample()
