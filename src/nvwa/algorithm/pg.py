from typing import Literal, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from nvwa.algorithm.base import Algorithm
from nvwa.data.batch import ReturnsBatch, RolloutBatch
from nvwa.distributions import (
    CategoricalDistribution,
    Distribution,
    GaussianDistribution,
)


class PolicyGradientAlgo(Algorithm):
    def __init__(
        self,
        actor: nn.Module,
        action_space: gym.Space,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.001,
        distribuiton: Optional[Distribution] = None,
        discount_factor: float = 0.98,
        gae_lambda: float = 0.95,
        reward_normalization: bool = False,
        is_action_scaling: bool = False,
        action_bound_method: Optional[Literal["clip", "tanh"]] = "clip",
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        max_grad_norm: float = 0,
    ):
        super().__init__(
            action_space=action_space,
            discount_factor=discount_factor,
            observation_space=observation_space,
            is_action_scaling=is_action_scaling,
            action_bound_method=action_bound_method,
        )

        self.actor = actor
        self.actor.to(device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self._set_distribution(distribuiton)
        self.gae_lambda = gae_lambda
        self.reward_normalization = reward_normalization
        self.dtype = dtype
        self.device = device
        self.max_grad_norm = max_grad_norm

    def _set_distribution(self, distribution: Optional[Distribution]) -> None:
        if distribution is None:
            if self.action_type == "discrete":
                self.distribution = CategoricalDistribution(self.action_space)
            elif self.action_type == "continuous":
                self.distribution = GaussianDistribution(self.action_space)
        else:
            self.distribution = distribution

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(observation)

    def get_action(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        latent = self.actor(observation)
        self.distribution.prob_distribution(latent)
        return self.distribution.get_action(deterministic)

    def get_log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = self.actor(observation)
        self.distribution.prob_distribution(latent)
        return self.distribution.log_prob(action)

    def compute_value(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def evaluate_observation(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.actor(observation)
        self.distribution.prob_distribution(logits=latent)
        action = self.distribution.get_action()
        log_prob = self.distribution.log_prob(action)
        return action, torch.zeros_like(log_prob), log_prob

    def learn(self, batch: ReturnsBatch) -> None:
        self.optimizer.zero_grad()
        for key in batch.keys():
            batch.get(key).requires_grad = True
        loss = -(batch.log_prob * batch.returns).mean()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": loss.item()}

    def process_rollout(self, batch: RolloutBatch) -> ReturnsBatch:
        returns, _ = self.compute_episode_return(
            batch, gamma=self.discount_factor, gae_lambda=self.gae_lambda
        )
        with torch.no_grad():
            log_prob = self.get_log_prob(
                self.wrapper.wrap_observation(batch.observation),
                self.wrapper.wrap_action(batch.action),
            ).unsqueeze(-1)
        return ReturnsBatch(
            observation=batch.observation,
            action=batch.action,
            reward=batch.reward,
            observation_next=batch.observation_next,
            terminated=batch.terminated,
            truncated=batch.truncated,
            returns=returns,
            log_prob=self.wrapper.to_numpy(log_prob),
        )
