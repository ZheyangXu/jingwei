import copy
from math import isnan
from typing import Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.agent.npg import NPG
from nvwa.data.batch import Batch
from nvwa.distributions import Distribution


class TRPO(NPG):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space: gym.Space,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.001,
        distribution: Optional[Distribution] = None,
        max_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        optimizing_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        normalize_advantages: bool = False,
        max_backtracks: int = 10,
        discount_factor: float = 0.98,
        gae_lambda: float = 0.95,
        max_batch_size: int = 256,
        reward_normalization: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        is_action_continuous: bool = False,
        is_action_scaling: bool = False,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            action_space=action_space,
            observation_space=observation_space,
            learning_rate=learning_rate,
            distribution=distribution,
            optimizing_critic_iters=optimizing_critic_iters,
            actor_step_size=actor_step_size,
            normalize_advantages=normalize_advantages,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
            max_batch_size=max_batch_size,
            reward_normalization=reward_normalization,
            device=device,
            dtype=dtype,
            is_action_continuous=is_action_continuous,
            is_action_scaling=is_action_scaling,
        )
        self.max_kl = max_kl
        self.max_batch_size = max_batch_size
        self.backtrack_coeff = backtrack_coeff
        self.max_backtracks = max_backtracks

    def learn(self, buffer: Batch) -> Dict[str, float]:
        epoch_loss = {"actor_loss": 0, "critic_loss": 0, "loss": 0}
        for step in range(self.max_gradient_step):
            for batch in buffer.get_batch(self.batch_size):
                with torch.no_grad():
                    old_logits = self.actor(batch.observation)
                    old_dist = torch.distributions.Categorical(logits=old_logits)
                logits = self.actor(batch.observation)
                self.distribution.prob_distribution(logits)
                dist = self.distribution.distribution
                ratio = (
                    (self.get_log_prob(batch.observation, batch.action) - batch.old_log_prob)
                    .exp()
                    .float()
                )
                actor_loss = -(ratio * batch.advantage).mean()
                flat_grads = self._get_flat_grad(actor_loss, self.actor, retain_graph=True).detach()
                kl = torch.distributions.kl_divergence(old_dist, dist).mean() + 1e-8
                flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
                search_direction = -self._conjugate_gradient(flat_grads, flat_kl_grad, max_iter=10)

                step_size = torch.sqrt(
                    2
                    * self.max_kl
                    / (
                        search_direction
                        * self._matrix_vector_product(search_direction, flat_kl_grad)
                    ).sum(0, keepdim=True)
                )
                with torch.no_grad():
                    flat_prams = torch.cat([pram.data.view(-1) for pram in self.actor.parameters()])

                    for i in range(self.max_backtracks):
                        new_flat_prams = flat_prams + step_size * search_direction
                        self._set_from_flat_params(self.actor, new_flat_prams)
                        new_logits = self.actor(batch.observation)
                        self.distribution.prob_distribution(new_logits)
                        new_dist = self.distribution.distribution
                        new_dratio = (
                            (new_dist.log_prob(batch.action) - batch.old_log_prob).exp().float()
                        )
                        new_actor_loss = -(new_dratio * batch.advantage).mean()
                        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
                        if kl < self.max_kl and new_actor_loss < actor_loss:
                            break
                        if i < self.max_backtracks - 1:
                            step_size *= self.backtrack_coeff
                        else:
                            self._set_from_flat_params
                            step_size = torch.tensor([0.0])

                for _ in range(self.optimizing_critic_iters):
                    value = self.critic(batch.observation)
                    value_loss = F.mse_loss(value, batch.returns)
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()

        return epoch_loss
