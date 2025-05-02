from abc import ABC, abstractmethod

import torch

from nvwa.data.batch import Batch


class Algorithm(ABC):

    @abstractmethod
    def get_action(self, observation: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def update(self, batch: Batch) -> None: ...


class OffPolicyAlgorithm(ABC, Algorithm):
    @abstractmethod
    def get_action_from_behavior(self, observation: torch.Tensor) -> torch.Tensor: ...
