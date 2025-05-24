from copy import deepcopy
from typing import Any, Dict, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.agent.a2c import ActorCritic
from nvwa.data.batch import Batch
from nvwa.data.buffer import RolloutBuffer
from nvwa.distributions import Distribution


class NPG(ActorCritic):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space: gym.Space,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.001,
        distribution: Optional[Distribution] = None,
        optimizing_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        normalize_advantages: bool = False,
        discount_factor: float = 0.98,
        max_gradient_step: int = 5,
        batch_size: int = 256,
        gae_lambda: float = 0.95,
        max_batch_size: int = 256,
        reward_normalization: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        is_action_continuous: bool = False,
        is_action_scaling: bool = False,
        action_bound_method: Optional[Literal["clip", "tanh"]] = "clip",
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            action_space=action_space,
            observation_space=observation_space,
            learning_rate=learning_rate,
            distribution=distribution,
            discount_factor=discount_factor,
            max_gradient_step=max_gradient_step,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
            entropy_coef=0.0,
            vf_coef=0.0,
            normalize_advantages=normalize_advantages,
            max_grad_norm=0.5,
            device=device,
            dtype=dtype,
            is_action_continuous=is_action_continuous,
            is_action_scaling=is_action_scaling,
            action_bound_method=action_bound_method,
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.max_batch_size = max_batch_size
        self.actor_step_size = actor_step_size
        self.optimizing_critic_iters = optimizing_critic_iters
        self.reward_normalization = reward_normalization
        self._dumping = 0.1

    def _get_flat_grad(self, y: torch.Tensor, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        grads = torch.autograd.grad(y, model.parameters(), **kwargs)
        return torch.cat([grad.reshape(-1) for grad in grads])

    def _matrix_vector_product(
        self, vector: torch.Tensor, flat_kl_grad: torch.Tensor
    ) -> torch.Tensor:
        kl_vector = (flat_kl_grad * vector).sum()
        flat_kl_grad_grad = self._get_flat_grad(kl_vector, self.actor, retain_graph=True).detach()
        return flat_kl_grad_grad + vector * self._dumping

    def _set_from_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        return model

    def _conjugate_gradient(
        self,
        vector: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        max_iter: int = 10,
        residual_tol: float = 1e-10,
    ) -> torch.Tensor:
        x = torch.zeros_like(vector)
        r = vector.clone()
        p = vector.clone()
        rdotr = r.dot(r)

        for _ in range(max_iter):
            z = self._matrix_vector_product(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def process_rollout(self, batch: Batch) -> Batch:
        with torch.no_grad():
            value = self.wrapper.to_numpy(
                self.compute_value(self.wrapper.wrap_observation(batch.observation))
            )
            value_next = self.wrapper.to_numpy(
                self.compute_value(self.wrapper.wrap_observation(batch.observation_next))
            )
            returns, advantage = self.compute_episode_return(
                batch,
                values=value,
                values_next=value_next,
                gamma=self.discount_factor,
                gae_lambda=self.gae_lambda,
            )
            logits = self.actor(self.wrapper.wrap_observation(batch.observation))
            self.distribution.prob_distribution(logits)
            dist = deepcopy(self.distribution.distribution)
            old_log_prob = self.distribution.log_prob(self.wrapper.wrap_action(batch.action))
            old_log_prob = self.wrapper.to_numpy(old_log_prob).reshape(-1, 1)
            batch.returns = returns
            batch.advantage = advantage
            batch.old_log_prob = old_log_prob
            batch.dist = dist
        return batch

    def learn(self, buffer: RolloutBuffer) -> Dict[str, float]:
        epoch_loss = {"actor_loss": 0, "critic_loss": 0, "loss": 0}
        for step in range(self.max_gradient_step):
            for batch in buffer.get_batch(self.batch_size):
                with torch.no_grad():
                    old_logits = self.actor(batch.observation)
                    old_dist = torch.distributions.Categorical(logits=old_logits)
                logits = self.actor(batch.observation)
                self.distribution.prob_distribution(logits)
                dist = self.distribution.distribution
                log_prob = self.get_log_prob(batch.observation, batch.action)
                actor_loss = -(log_prob * batch.advantage).mean()
                self.optimizer.zero_grad()
                actor_loss.backward()
                flat_grads = self._get_flat_grad(actor_loss, self.actor, retain_graph=True).detach()

                kl = torch.distributions.kl_divergence(old_dist, dist).mean()
                flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
                search_direction = -self._conjugate_gradient(flat_grads, flat_kl_grad, max_iter=10)

                with torch.no_grad():
                    flat_prams = torch.cat([param.view(-1) for param in self.actor.parameters()])
                    new_flat_params = flat_prams + self.actor_step_size * search_direction
                    self._set_from_flat_params(self.actor, new_flat_params)
                    new_logits = self.actor(batch.observation)
                    self.distribution.prob_distribution(new_logits)
                    # new_dist = self.distribution.distribution

                total_value_loss = 0.0
                for _ in range(self.optimizing_critic_iters):
                    value = self.compute_value(batch.observation)
                    value_loss = F.mse_loss(value, batch.returns)
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()
                    total_value_loss += value_loss.item()
                    epoch_loss["critic_loss"] += value_loss.item()
                epoch_loss["actor_loss"] += actor_loss.item()
                epoch_loss["loss"] += (
                    actor_loss.item() + total_value_loss / self.optimizing_critic_iters
                )
        return epoch_loss
