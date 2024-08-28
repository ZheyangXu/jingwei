from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from jingwei.domain.actor.base import BaseActor
from jingwei.domain.agent.base import BaseAgent
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

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(observation)
    
    # TODO: add critic for dqn?
    def estimate_return(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        return super().estimate_return(transitions=transitions)

    def update_step(self, transitions: TensorTransitionBatch, step: int = 1) -> None:
        loss = self.compute_actor_loss(transitions)
        self.actor.update_step(loss)
        if step % self.target_update_step == 0:
            self.target_actor = deepcopy(self.actor)

    def compute_actor_loss(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        q_values = self.actor.get_probs(transitions.observation).max(dim=1)[0]
        max_next_q_values = self.target_actor.get_probs(transitions.observation_next).max(dim=1)[0]
        q_targets = transitions.reward + self.gamma * max_next_q_values * (
            1 - transitions.terminated
        )
        return torch.mean(F.mse_loss(q_values, q_targets))

    def mtype(self) -> MType:
        return MType.off_policy
