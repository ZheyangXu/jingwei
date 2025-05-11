from abc import ABC, abstractmethod
from typing import Tuple

import gymnasium as gym
import torch


class Distribution(ABC):
    distribution: torch.distributions.Distribution

    def __init__(self, action_space: gym.spaces.Space) -> None:
        self.action_space = action_space

    @abstractmethod
    def prob_distribution(self, logits: torch.Tensor | Tuple[torch.Tensor, ...]) -> None: ...

    @abstractmethod
    def log_prob(self, action: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def entropy(self) -> torch.Tensor: ...

    @abstractmethod
    def sample(self, batch_size: int) -> torch.Tensor: ...

    @abstractmethod
    def mode(self) -> torch.Tensor: ...

    def get_action(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()
