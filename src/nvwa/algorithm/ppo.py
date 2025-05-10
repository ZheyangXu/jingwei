import dis
from operator import is_
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.algorithm.a2c import ActorCritic
from nvwa.data.batch import RolloutBatch


class PPO(ActorCritic):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        learning_rate: float = 0.001,
        distribution: Optional[torch.distributions.Distribution] = None,
        n_epochs: int = 2,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.0,
        vf_coef: float = 0.5,
        normalize_advantages: bool = True,
        lmbda: float = 0.9,
        eps: float = 0.2,
        max_grad_norm: float = 0.5,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        is_action_continuous: bool = False,
        action_scaling: bool = False,
        action_bound_method: Optional[str] = "clip",
    ) -> None:
        super(PPO, self).__init__(
            actor,
            critic,
            learning_rate,
            distribution,
            discount_factor,
            gae_lambda,
            entropy_coef,
            vf_coef,
            normalize_advantages,
            max_grad_norm,
            device,
            dtype,
            is_action_continuous,
            action_scaling,
            action_bound_method,
        )
        self.lmbda = lmbda
        self.eps = eps
        self.n_epochs = n_epochs

    def update(self, batch: RolloutBatch) -> None:
        values, log_probs, entropy = self.evaluate_action(batch.observation, batch.action)
        advantages = batch.advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - batch.log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        value_loss = F.mse_loss(batch.returns, values)

        if entropy is None:
            entropy_loss = -torch.mean(-log_probs)
        else:
            entropy_loss = -torch.mean(entropy)

        loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": policy_loss.item(),
            "critic_loss": value_loss.item(),
            "loss": loss.item(),
            "entropy_loss": entropy_loss.item(),
        }
