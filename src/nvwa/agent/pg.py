from typing import Literal, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from nvwa.agent.base import BaseAgent
from nvwa.data.batch import Batch
from nvwa.data.buffer import RolloutBuffer
from nvwa.distributions import (
    CategoricalDistribution,
    Distribution,
    GaussianDistribution,
)


class PolicyGradientAlgo(BaseAgent):
    def __init__(
        self,
        actor: nn.Module,
        action_space: gym.Space,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.001,
        distribuiton: Optional[Distribution] = None,
        discount_factor: float = 0.98,
        max_gradient_step: int = 5,
        batch_size: int = 256,
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
        self.max_gradient_step = max_gradient_step
        self.batch_size = batch_size

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

    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
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

    def learn(self, buffer: RolloutBuffer) -> None:
        epoch_loss = {"loss": 0, "actor_loss": 0, "critic_loss": 0}
        for _ in range(self.max_gradient_step):
            for batch in buffer.get_batch(self.batch_size):
                log_prob = self.get_log_prob(batch.observation, batch.action)
                loss = -(log_prob * batch.returns).mean()
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                epoch_loss["loss"] += loss.item()
        return epoch_loss

    def process_rollout(self, batch: Batch) -> Batch:
        returns, _ = self.compute_episode_return(
            batch, gamma=self.discount_factor, gae_lambda=self.gae_lambda
        )
        batch.returns = returns
        return batch
