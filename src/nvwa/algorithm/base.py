from abc import ABC, abstractmethod
from typing import Tuple

import torch

from nvwa.data.batch import Batch


class Algorithm(ABC):

    @abstractmethod
    def get_action(self, observation: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def update(self, batch: Batch) -> None: ...


class OffPolicyAlgorithm(Algorithm):
    @abstractmethod
    def get_behavior_action(self, observation: torch.Tensor) -> torch.Tensor: ...


class OnPolicyAlgorithm(Algorithm):
    @abstractmethod
    def evaluate_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def estimate_value(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
