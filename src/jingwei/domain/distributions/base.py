from abc import ABC, abstractmethod
from typing import TypeVar

from jingwei.infra.typing import *


DistributionType = TypeVar("DistributionType", bound="Distribution")


class Distribution(ABC):
    @abstractmethod
    def proba_distrbution_net():
        pass

    @abstractmethod
    def prob_distribution(distribution: DistributionType) -> DistributionType:
        pass

    @abstractmethod
    def log_prob(observation: ObservationType) -> TensorType:
        pass

    @abstractmethod
    def entropy() -> TensorType:
        pass

    @abstractmethod
    def sample() -> ActionType:
        pass

    @abstractmethod
    def mode() -> ActionType:
        pass

    @abstractmethod
    def get_action() -> ActionType:
        pass
