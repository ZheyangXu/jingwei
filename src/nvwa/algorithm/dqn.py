import random

import gymnasium as gym
import torch
import torch.nn.functional as F

from nvwa.actor.q_actor import QActor
from nvwa.algorithm.off_policy import OffPolicyAlgorithm
from nvwa.data.batch import Batch


class DQN(OffPolicyAlgorithm):
    def __init__(
        self,
        actor: QActor,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        gamma: float = 0.99,
        target_update_step: int = 2,
        epsilon: float = 0.1,
    ) -> None:
        self.actor = actor
        self.target_actor = actor.clone()
        self.target_actor.is_target_actor = True
        self.epsilon = epsilon
        self.observation_space = observation_space
        self.action_space = action_space
        self.target_update_step = target_update_step
        self.gamma = gamma
        self.global_step = 0

    def get_behavior_action(self, observation: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) < self.epsilon:
            return torch.tensor(self.action_space.sample())
        else:
            return self.actor.get_action(observation)

    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor.get_action(observation)

    def update(self, batch: Batch) -> None:
        q_values = self.actor.get_max_q_values(batch.observation, batch.action)
        max_next_q_values = self.target_actor.get_max_q_values(batch.observation_next)

        q_targets = batch.reward + self.gamma * max_next_q_values * (
            1 - torch.logical_or(batch.terminated, batch.truncated).float()
        )

        loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.actor.update_step(loss)
        self.global_step += 1
        if self.global_step % self.target_update_step == 0:
            self.target_actor.update_target(self.actor)
        return {
            "loss": loss.item(),
            "q_values": q_values.mean().item(),
            "q_targets": q_targets.mean().item(),
        }

    def is_off_policy(self) -> bool:
        return True

    def is_offline(self) -> bool:
        return False
