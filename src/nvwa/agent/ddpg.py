from copy import deepcopy
from typing import Dict, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nvwa.agent.base import BaseAgent
from nvwa.data.batch import Batch
from nvwa.data.buffer import ReplayBuffer


class DDPG(BaseAgent):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space: gym.Space,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.01,
        sigma: float = 0.1,
        tau: float = 0.005,
        gamma: float = 0.99,
        estimate_step: int = 1,
        is_action_scaling: bool = False,
        action_bound_method: Optional[Literal["clip"]] = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            discount_factor=gamma,
            is_action_scaling=is_action_scaling,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.actor = actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.target_actor = deepcopy(actor)
        self.target_actor.eval()
        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.target_critic = deepcopy(critic)
        self.target_critic.eval()
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.estimate_step = estimate_step
        self.is_action_scaling = is_action_scaling
        self.action_dim = action_space.shape[0]
        self.action_bound_method = action_bound_method

    def get_action(self, observation, deterministic=False) -> torch.Tensor:
        action = self.actor(observation)
        return self.exploration_noise(action)

    def get_behavior_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.get_action(observation, deterministic=False)

    def exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        return action + self.sigma * torch.randn(self.action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

    def learn(self, batch: Batch) -> Dict[str, float]:
        epoch_loss = {"actor_loss": 0.0, "critic_loss": 0.0, "loss": 0.0}
        next_q_value = self.target_critic(
            batch.observation_next, self.target_actor(batch.observation_next)
        )
        q_target = batch.reward + self.gamma * next_q_value * (
            1 - torch.logical_or(batch.terminated, batch.truncated).float()
        )
        critic_loss = torch.mean(F.mse_loss(self.critic(batch.observation, batch.action), q_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(batch.observation, self.actor(batch.observation)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        epoch_loss["actor_loss"] = actor_loss.item()
        epoch_loss["critic_loss"] = critic_loss.item()
        epoch_loss["loss"] = actor_loss.item() + critic_loss.item()
        return epoch_loss

    def soft_update(self, model: nn.Module, target_model: nn.Module) -> None:
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def process_rollout(self, batch: Batch) -> Batch:
        return batch
