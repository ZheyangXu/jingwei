import random
import torch
import torch.nn.functional as F
import gymnasium as gym

from nvwa.actor.q_actor import QActor
from nvwa.data.batch import Batch
from nvwa.algorithm.base import Algorithm


class DQN(Algorithm):
    def __init__(
        self,
        actor: QActor,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        gamma: float,
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

    def get_action_from_behavior(self, observation: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) < self.epsilon:
            self.action_space.sample()
        else:
            self.actor.get_action(observation)
    
    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(observation)

    def update(self, batch: Batch) -> None:
        q_values = self.actor.get_q_values(batch.observation)
        max_next_q_values = self.target_actor.get_max_q_values(batch.observation_next)
        q_targets = batch.reward + self.gamma * max_next_q_values * (1 - batch.terminated)

        loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.actor.update_step(loss)
        self.global_step += 1
        if self.global_step % self.target_update_step == 0:
            self.target_actor.update_target(self.actor)

    def is_off_policy(self) -> bool:
        return True

    def is_offline(self) -> bool:
        return False
