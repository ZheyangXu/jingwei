from typing import Tuple

import gymnasium as gym
import torch

from nvwa.distributions.base import Distribution


class GaussianDistribution(Distribution):
    def __init__(self, action_space: gym.spaces.Box) -> None:
        self.action_space = action_space
        self.action_dimension = action_space.shape[0]

    @staticmethod
    def _sum_independet_dims(tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor) > 1:
            return tensor.sum(dim=1)
        return tensor.sum()

    def prob_distribution(self, logits: Tuple[torch.Tensor, torch.Tensor]) -> None:
        mu, std = logits
        self.distribution = torch.distributions.Normal(mu, std)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(action)
        return self._sum_independet_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        entropy = self.distribution.entropy()
        return self._sum_independet_dims(entropy)

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def probs(self) -> torch.Tensor:
        return self.distribution.mean
