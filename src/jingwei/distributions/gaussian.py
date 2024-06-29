from typing import Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch.distributions import Normal

from jingwei.domain.distributions.base import Distribution
from jingwei.infra.typing import *

DiagGaussianDistributionType = TypeVar("DiagGaussianDistributionType", bound="DiagGaussianDistribution")
SquashedDiagGaussianDistributionType = TypeVar(
    "SquashedDiagGaussianDistributionType", bound="SquashedDiagGaussianDistribution"
)


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,) for (n_batch, n_actions) input, scalar for (n_batch,) input
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    def __init__(self) -> None:
        super().__init__()

    def prob_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> DiagGaussianDistributionType:
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(action)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def get_action(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gaussian_action: Optional[torch.Tensor] = None

    def prob_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> DiagGaussianDistributionType:
        super().prob_distribution(mean_actions, log_std)
        return self

    def log_prob(self, action: torch.Tensor, gaussian_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if gaussian_action is None:
            gaussian_action = TanhBijector.inverse(action)

        log_prob = super().log_prob(gaussian_action)
        log_prob -= torch.sum(torch.log(1 - action**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        return None

    def sample(self) -> torch.Tensor:
        self.gaussian_action = super().sample()
        return torch.tanh(self.gaussian_action)

    def mode(self) -> torch.Tensor:
        self.gaussian_action = super().mode()
        return torch.tanh(self.gaussian_action)


class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(y.dtype).eps
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)
