from abc import ABC, abstractmethod

import torch

from nvwa.algorithm.base import Algorithm


class OffPolicyAlgorithm(Algorithm, ABC):

    @abstractmethod
    def get_behavior_action(self, observation: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_max_q_values(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor: ...

    def is_off_policy(self) -> bool:
        return True

    def is_on_policy(self) -> bool:
        return False

    def is_offline(self) -> bool:
        return False
