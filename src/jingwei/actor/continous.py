# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from jingwei.actor.base import BaseActor
from jingwei.infra.typing import ActionType, StateType


class ContinuousActor(BaseActor):
    def __init__(self, model: nn.Module, optimzier: optim.Optimizer) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimzier

    def take_action(self, state: StateType) -> ActionType:
        mu, sigma = self.model(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action

    def update_fn(self, loss: type[float] | torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad(True)
        loss.backward()
        self.optimizer.step()

    def get_log_probs(self, states: StateType, actions: ActionType) -> torch.Tensor:
        mu, std = self.model(states)
        action_dist = torch.distributions.Normal(mu.detach(), std.detach())
        return action_dist.log_prob(actions)
