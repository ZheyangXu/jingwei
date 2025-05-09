from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nvwa.algorithm.base import OnPolicyAlgorithm
from nvwa.data.batch import RolloutBatch


class ActorCritic(nn.Module, OnPolicyAlgorithm):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        learning_rate: float = 0.001,
        distribution: Optional[torch.distributions.Distribution] = None,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.0,
        vf_coef: float = 0.5,
        normalize_advantages: bool = False,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.normalize_advantages = normalize_advantages
        self.device = device
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self._set_distribution(distribution)

    def _set_distribution(self, distribution: Optional[torch.distributions.Distribution]) -> None:
        if distribution is None:
            self.distribution = torch.distributions.Categorical
        else:
            self.distribution = distribution

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.critic(observation)
        latent = self.actor(observation)
        dist = self.distribution(logits=latent)
        action = dist.sample()
        log_prob = dist.log_prob(action.squeeze(-1).long())
        return action, value, log_prob

    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        latent = self.actor(observation)
        dist = self.distribution(logits=latent)
        if deterministic:
            action = torch.argmax(dist.probs, dim=1)
        else:
            action = dist.sample()
        return action

    def estimate_value(self, observation) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(observation)

    def get_log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = self.actor(observation)
        dist = self.distribution(logits=latent)
        log_prob = dist.log_prob(action.long())
        return log_prob

    def compute_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def evaluate_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.actor(observation)
        dist = self.distribution(logits=latent)
        log_prob = dist.log_prob(action.squeeze(-1).long())
        value = self.compute_value(observation)
        return value, log_prob.view(-1, 1), dist.entropy().view(-1, 1)

    def update(self, batch: RolloutBatch) -> None:
        values, log_prob, entropy = self.evaluate_action(batch.observation, batch.action)
        advantages = batch.advantages
        returns = batch.returns
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = -(log_prob * advantages).mean()
        value_loss = F.mse_loss(returns, values).mean()

        entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)

        loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        loss.backward()
        self.optimizer.step()

        return {
            "actor_loss": policy_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item(),
        }
