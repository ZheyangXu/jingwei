import copy
import re
from typing import Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.algorithm.npg import NPG
from nvwa.data.batch import AdvantageBatch
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

    def hessian_matrix_vector_product(
        self,
        batch: AdvantageBatch,
        old_dist: torch.distributions.Distribution,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        self.distribution.prob_distribution(self.actor(batch.observation))
        new_dist = self.distribution.distribution
        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), retain_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(
            kl_grad_vector_product, self.actor.parameters(), retain_graph=True
        )
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(
        self,
        grad: torch.Tensor,
        batch: AdvantageBatch,
        old_dist: torch.distributions.Distribution,
    ) -> torch.Tensor:
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for _ in range(10):
            Hp = self.hessian_matrix_vector_product(batch, old_dist, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_objective(self, batch: AdvantageBatch, actor: nn.Module) -> torch.Tensor:
        logits = actor(batch.observation)
        self.distribution.prob_distribution(logits)
        log_probs = self.distribution.log_prob(batch.action)
        ratio = torch.exp(log_probs - batch.old_log_prob)
        return torch.mean(ratio * batch.advantage)

    def line_search(
        self,
        batch: AdvantageBatch,
        old_dist: torch.distributions.Distribution,
        max_vector: torch.Tensor,
    ) -> torch.ParameterDict:
        old_para = nn.utils.convert_parameters(self.actor.parameters())
        old_obj = self.compute_surrogate_objective(batch, self.actor)
        for i in range(self.max_backtrack_iters):
            coef = self.actor_step_size**i
            new_para = old_para + coef * max_vector
            new_actor = copy.deepcopy(self.actor)
            nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            self.distribution.prob_distribution(new_actor(batch.observation))
            new_dist = self.distribution.distribution
            kl_div = torch.mean(torch.distributions.kl_divergence(old_dist, new_dist))
            new_obj = self.compute_surrogate_objective(batch, new_actor)
            if new_obj > old_obj and kl_div < self.max_kl:
                return new_para
        return old_para

    def policy_learn(
        self, batch: AdvantageBatch, old_dist: torch.distributions.Distribution
    ) -> Dict[str, float]:
        surrogate_objective = self.compute_surrogate_objective(batch, self.actor)
        grads = torch.autograd.grad(surrogate_objective, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        descent_direction = self.conjugate_gradient(obj_grad, batch, old_dist)
        Hd = self.hessian_matrix_vector_product(batch, old_dist, descent_direction)
        max_coef = torch.sqrt(2 * self.max_kl / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(batch, old_dist, descent_direction * max_coef)
        nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def learn(self, batch: AdvantageBatch) -> Dict[str, float]:
        td_target = batch.reward + self.gamma * self.compute_value(batch.observation_next) * (
            1 - torch.logical_or(batch.terminated, batch.truncated).float()
        )
        td_delta = td_target - self.compute_value(batch.observation)
        self.distribution.prob_distribution(self.actor(batch.observation))
        old_dist = self.distribution.distribution
        critic_loss = torch.mean(F.mse_loss(td_target, self.compute_value(batch.observation)))
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()
        self.policy_learn(batch, old_dist)
        return {
            "loss": critic_loss.item(),
            "actor_loss": 0.0,
            "critic_loss": critic_loss.item(),
        }
