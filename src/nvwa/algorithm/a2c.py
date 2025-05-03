from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from nvwa.data.batch import RolloutBatch
from nvwa.data.transition import RolloutTransition
from nvwa.data.buffer import RolloutBuffer
from nvwa.actor.actor import Actor
from nvwa.critic.base import Critic
from nvwa.algorithm.base import OnPolicyAlgorithm


class ActorCritic(OnPolicyAlgorithm):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        gamma: float = 0.9,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(observation)

    def estimate_value(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.actor.get_action(observation)
        log_prob = self.actor.get_log_prob(observation, action)
        return action, self.critic.estimate_return(observation), log_prob

    def evaluate_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def update(self, batch: RolloutBatch) -> None:
        td_target = batch.reward + self.gamma * self.critic.estimate_return(
            batch.observation_next
        ) * (1 - torch.logical_or(batch.terminated, batch.truncated).float())
        td_delta = td_target - self.critic.estimate_return(batch.observation)
        
        log_probs = self.actor.get_log_prob(batch.observation, batch.action)
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic.estimate_return(batch.observation), td_target)
        )
        self.actor.update_step(actor_loss)
        self.critic.update_step(critic_loss)
