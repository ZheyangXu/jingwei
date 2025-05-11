import gymnasium as gym
import torch

from nvwa.distributions.base import Distribution
from nvwa.infra.functional import get_action_dimension


class CategoricalDistribution(Distribution):
    def __init__(self, action_space: gym.spaces.Discrete) -> None:
        super().__init__(action_space)
        self.action_space = action_space
        self.action_dimension = get_action_dimension(action_space)
        self.distribution = torch.distributions.Categorical

    def prob_distribution(self, logits: torch.Tensor) -> None:
        self.distribution = self.distribution(logits=logits)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(action.long())

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)
