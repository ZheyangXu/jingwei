from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn

from nvwa.data.batch import Batch
from nvwa.infra.functional import get_action_dimension, get_action_type


class Algorithm(nn.Module, ABC):

    def __init__(
        self,
        *,
        action_space: gym.Space,
        discount_factor: float = 0.99,
        observation_space: Optional[gym.Space] = None,
        is_action_scaling: bool = False,
        action_bound_method: Optional[str] = "clip",
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dimension = get_action_dimension(action_space)
        self.is_action_scaling = is_action_scaling
        self.action_bound_method = action_bound_method
        self.action_type = get_action_type(action_space)
        self.discount_factor = discount_factor

    @abstractmethod
    def forward(
        self, observation: torch.Tensor, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def get_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor: ...

    def map_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_type == "continuous":
            with torch.no_grad():
                if self.action_bound_method == "clip":
                    action = torch.clamp(action, -1.0, 1.0)
                elif self.action_bound_method == "tanh":
                    action = torch.tanh(action)
                if self.is_action_scaling:
                    low, high = self.action_space.low, self.action_space.high
                    action = low + (high - low) * (action + 1.0) / 2.0
        return action

    def map_action_inverse(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_type == "continuous":
            with torch.no_grad():
                if self.is_action_scaling:
                    low, high = self.action_space.low, self.action_space.high
                    action = 2.0 * (action - low) / (high - low) - 1.0
                if self.action_bound_method == "tanh":
                    action = (torch.log(1.0 + action) - torch.log(1.0 - action)) / 2.0
        return action

    @abstractmethod
    def learn(self, batch: Batch) -> None: ...
