from typing import List, TypeVar

import torch
import torch.nn as nn
from torch.distributions import Categorical

from jingwei.domain.distributions.base import Distribution
from jingwei.infra.typing import *

CategorialDistributionType = TypeVar("CategorialDistributionType", bound="CategorialDistribution")
MultiCategoricalDistributionType = TypeVar("MultiCategoricalDistributionType", bound="MultiCategoricalDistribution")


class CategorialDistribution(Distribution):
    def __init__(self) -> None:
        super().__init__()

    def prob_distribution(self, action_logits: torch.Tensor) -> CategorialDistributionType:
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)


class MultiCategoricalDistribution(Distribution):
    def __init__(self, action_dims: List[int]) -> None:
        super().__init__()
        self.action_dims = action_dims

    def prob_distribution(self, action_logits: torch.Tensor) -> MultiCategoricalDistributionType:
        self.distribution = [
            Categorical(logits=split_logit) for split_logit in torch.split(action_logits, list(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)
