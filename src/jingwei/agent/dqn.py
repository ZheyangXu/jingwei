from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from jingwei.domain.actor import BaseActor
from jingwei.domain.agent import BaseAgent
from jingwei.infra.mtype import MType
from jingwei.infra.typing import *
from jingwei.transitions.base import TensorTransitionBatch


class DQNAgent(BaseAgent):
    def __init__(self, actor: BaseActor, target_update_step: int = 2, gamma: float = 0.9) -> None:
        super().__init__()
        self.actor = actor
        self.target_actor = deepcopy(actor)
        self.target_update_step = target_update_step
        self.gamma = gamma
        self.step: int = 0

    def update_target_actor(self) -> None:
        self.step += 1
        if self.step % self.target_update_step == 0:
            self.target_actor.model.load_state_dict(self.actor.model.state_dict())

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(observation)

    # TODO: add critic for dqn?
    def estimate_return(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        return self.actor.get_values(transitions.observation)

    def update_step(self, transitions: TensorTransitionBatch) -> None:
        loss = self.compute_actor_loss(transitions)
        self.actor.update_step(loss)
        self.update_target_actor()

    def compute_actor_loss(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        q_values = self.actor.get_values(transitions.observation).max(dim=1).values.reshape((-1, 1))
        next_q_values = (
            self.target_actor.get_values(transitions.observation_next)
            .max(dim=1)
            .values.reshape((-1, 1))
        )
        q_targets = transitions.reward + self.gamma * next_q_values * (1 - transitions.terminated)
        return torch.mean(F.mse_loss(q_values, q_targets))

    def mtype(self) -> MType:
        return MType.off_policy
