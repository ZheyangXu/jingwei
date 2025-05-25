from copy import deepcopy
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nvwa.agent.base import BaseAgent
from nvwa.data.batch import Batch


class SAC(BaseAgent):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space: gym.Space,
        critic2: Optional[nn.Module] = None,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.01,
        tau: float = 0.005,
        sigma: float = 0.1,
        alpha: float = 0.2,
        gamma: float = 0.99,
        target_entropy: Optional[float] = None,
        discount_factor: float = 0.99,
        is_action_scaling: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device("cpu"),
    ):
        super().__init__(
            action_space=action_space,
            discount_factor=discount_factor,
            observation_space=observation_space,
            is_action_scaling=is_action_scaling,
            dtype=dtype,
            device=device,
        )
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.target_critic = deepcopy(critic)
        self.target_critic.eval()
        self.critic2 = critic2 if critic2 is not None else deepcopy(critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.target_critic2 = deepcopy(self.critic2)
        self.target_critic2.eval()
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.tau = tau
        self.sigma = sigma
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(action_space.shape[0])
        )
        self.gamma = gamma

    def get_action(self, observation: torch.Tensor, **kwargs) -> torch.Tensor:
        action, _ = self.actor(observation)
        return action

    def get_behavior_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.get_action(observation, deterministic=False)

    def forward(self, observation: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.actor(observation)

    def soft_update(self, model: nn.Module, target_model: nn.Module) -> None:
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def compute_target(self, batch: Batch) -> torch.Tensor:
        action_next, log_prob = self.actor(batch.observation_next)
        entropy = -log_prob
        q1 = self.target_critic(batch.observation_next, action_next)
        q2 = self.target_critic2(batch.observation_next, action_next)
        next_q = torch.min(q1, q2) - self.log_alpha.exp() * entropy
        td_target = batch.reward + self.discount_factor * next_q * (
            1 - torch.logical_or(batch.terminated, batch.truncated).float()
        )
        return td_target

    def learn(self, batch: Batch) -> dict[str, float]:
        epoch_loss = {"actor_loss": 0.0, "critic_loss": 0.0, "loss": 0.0}
        td_target = self.compute_target(batch)
        critic1_loss = torch.mean(
            F.mse_loss(self.critic(batch.observation, batch.action), td_target.detach())
        )
        critic2_loss = torch.mean(
            F.mse_loss(self.critic2(batch.observation, batch.action), td_target.detach())
        )
        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        action, log_prob = self.actor(batch.observation)
        entropy = -log_prob
        q1 = self.critic(batch.observation, action)
        q2 = self.critic2(batch.observation, action)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy + torch.min(q1, q2))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.critic2, self.target_critic2)
        epoch_loss["actor_loss"] = actor_loss.item()
        epoch_loss["critic_loss"] = critic1_loss.item() + critic2_loss.item()
        epoch_loss["loss"] = actor_loss.item() + critic1_loss.item() + critic2_loss.item()
        return epoch_loss

    def process_rollout(self, batch: Batch) -> Batch:
        return batch
