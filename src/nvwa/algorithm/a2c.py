from typing import Literal, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nvwa.algorithm.base import Algorithm
from nvwa.data.batch import AdvantageBatch, RolloutBatch
from nvwa.distributions import (
    CategoricalDistribution,
    Distribution,
    GaussianDistribution,
)


class ActorCritic(Algorithm):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space: gym.Space,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.001,
        distribution: Optional[Distribution] = None,
        discount_factor: float = 0.98,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.0,
        vf_coef: float = 0.5,
        normalize_advantages: bool = False,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        is_action_continuous: bool = False,
        is_action_scaling: bool = False,
        action_bound_method: Optional[Literal["clip", "tanh"]] = "clip",
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            discount_factor=discount_factor,
            is_action_scaling=is_action_scaling,
            action_bound_method=action_bound_method,
            dtype=dtype,
            device=device,
        )
        self.actor = actor
        self.critic = critic
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.normalize_advantages = normalize_advantages
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.is_action_continuous = is_action_continuous
        self.action_bound_method = action_bound_method
        self._set_distribution(distribution)

    def _set_distribution(self, distribution: Optional[torch.distributions.Distribution]) -> None:
        if distribution is None:
            if self.action_type == "discrete":
                self.distribution = CategoricalDistribution(self.action_space)
            elif self.action_type == "continuous":
                self.distribution = GaussianDistribution(self.action_space)
        else:
            self.distribution = distribution

    def _action_bound(self, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.action_bound_method == "clip":
                action = torch.clamp(action, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                action = torch.tanh(action)
        return action

    def _action_scaling(self, action: torch.Tensor, low: float, high: float) -> torch.Tensor:
        with torch.no_grad():
            if self.action_scaling:
                action = low + (high - low) * (action + 1.0) / 2.0
        return action

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.critic(observation)
        latent = self.actor(observation)
        self.distribution.prob_distribution(latent)
        action = self.distribution.get_action()
        log_prob = self.distribution.log_prob(action)
        return action, value, log_prob

    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        latent = self.actor(observation)
        self.distribution.prob_distribution(latent)
        action = self.distribution.get_action(deterministic)
        return action

    def get_log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = self.actor(observation)
        self.distribution.prob_distribution(latent)
        log_prob = self.distribution.log_prob(action)
        return log_prob

    def compute_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def evaluate_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.actor(observation)
        self.distribution.prob_distribution(latent)
        log_prob = self.distribution.log_prob(action)
        entropy = self.distribution.entropy()
        value = self.compute_value(observation)
        return value, log_prob.view(-1, 1), entropy.view(-1, 1)

    def learn(self, batch: AdvantageBatch) -> None:
        values, log_prob, entropy = self.evaluate_action(batch.observation, batch.action)
        advantages = batch.advantage
        returns = batch.returns
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = -(log_prob * advantages).mean()
        value_loss = F.mse_loss(returns, values).mean()

        entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)

        loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        loss.backward()
        self.optimizer.step()

        return {
            "actor_loss": policy_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item(),
        }

    def process_rollout(self, batch: RolloutBatch) -> AdvantageBatch:
        returns, advantages = self.compute_episode_return(
            batch, gamma=self.discount_factor, gae_lambda=self.gae_lambda
        )

        return AdvantageBatch(
            observation=batch.observation,
            action=batch.action,
            reward=batch.reward,
            observation_next=batch.observation_next,
            terminated=batch.terminated,
            truncated=batch.truncated,
            advantages=advantages,
            returns=returns,
            old_log_prob=None,
        )

    def to(self, device: torch.device) -> None:
        self._device = device
        self.actor.to(device)
        self.critic.to(device)
