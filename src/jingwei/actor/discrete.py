# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from jingwei.actor.base import BaseActor
from jingwei.infra.typing import ActionType, StateType


class DiscreteActor(BaseActor):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def take_action(self, state: StateType) -> ActionType:
        probs = self.model(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action

    def update_fn(self, loss: type[float] | torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad(True)
        loss.backward()
        self.optimizer.step()

    def get_log_probs(self, states: StateType, actions: ActionType) -> torch.Tensor:
        return torch.log(self.model(states).gather(1, actions)).detach()
