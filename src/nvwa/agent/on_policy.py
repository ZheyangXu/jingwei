from abc import ABC, abstractmethod
from typing import Tuple

import torch

from nvwa.agent.base import Algorithm


class OnPolicyAlgorithm(Algorithm, ABC):

    @abstractmethod
    def evaluate_observation(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def compute_value(self, observation: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor: ...

    @abstractmethod
    def evaluate_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def is_off_policy(self) -> bool:
        return False

    def is_on_policy(self) -> bool:
        return True

    def is_offline(self) -> bool:
        return False
