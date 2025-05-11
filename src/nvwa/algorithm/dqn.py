import random
from copy import deepcopy
from typing import Any, Dict, List, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.algorithm.off_policy import OffPolicyAlgorithm
from nvwa.data.batch import Batch


class DQN(nn.Module, OffPolicyAlgorithm):
    def __init__(
        self,
        actor: nn.Module,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        is_double_dqn: bool = False,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_step: int = 2,
        epsilon: float = 0.1,
        tau: float = 0.05,
        is_soft_update: bool = False,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(DQN, self).__init__()
        self.actor = actor
        self.target_actor = deepcopy(actor)
        self.target_actor.eval()
        self.learning_rate = learning_rate
        self.is_double_dqn = is_double_dqn
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self._device = device
        self.dtype = dtype
        self.epsilon = epsilon
        self.observation_space = observation_space
        self.action_space = action_space
        self.target_update_step = target_update_step
        self.gamma = gamma
        self.tau = tau
        self.is_soft_update = is_soft_update
        self.global_step = 0

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        dist = self.actor(observation)
        return dist

    def get_behavior_action(self, observation: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) < self.epsilon:
            return torch.tensor(self.action_space.sample())
        else:
            return self.get_action(observation)

    def get_action(
        self, observation: torch.Tensor, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        action_dist = self.actor(observation)
        return action_dist.argmax()

    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

    def get_max_q_values(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        actor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if actor is None:
            actor = self.actor
        if action is not None:
            max_q_values = actor(observation).gather(1, action.long())
        else:
            max_q_values = actor(observation).max(1)[0].view(-1, 1)
        return max_q_values

    def update_target(self) -> None:
        self.target_actor.load_state_dict(self.actor.state_dict())

    def soft_update_target(self) -> None:
        self.target_actor.load_state_dict(
            self.target_actor.state_dict()
            + self.tau * (self.actor.state_dict() - self.target_actor.state_dict())
        )

    def update(self, batch: Batch) -> None:
        q_values = self.get_max_q_values(batch.observation, batch.action)
        if self.is_double_dqn:
            next_actions = self.actor(batch.observation_next).argmax(1, keepdim=True)
            max_next_q_values = self.get_max_q_values(
                batch.observation_next, action=next_actions, actor=self.target_actor
            )
        else:
            max_next_q_values = self.get_max_q_values(
                batch.observation_next, actor=self.target_actor
            )

        q_targets = batch.reward + self.gamma * max_next_q_values * (
            1 - torch.logical_or(batch.terminated, batch.truncated).float()
        )

        loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        if self.global_step % self.target_update_step == 0:
            if self.is_soft_update:
                self.soft_update_target()
            else:
                self.update_target()
        return {
            "loss": loss.item(),
            "q_values": q_values.mean().item(),
            "q_targets": q_targets.mean().item(),
        }

    def is_off_policy(self) -> bool:
        return True

    def is_on_policy(self):
        return False

    def is_offline(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Optional[torch.device] = None) -> None:
        device = device or self._device
        self.actor.to(device)
        self.target_actor.to(device)
        self._device = device
