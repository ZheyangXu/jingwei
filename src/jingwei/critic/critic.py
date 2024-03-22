# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from jingwei.critic.base import BaseCritic
from jingwei.infra.typing import StateType, ValueType


class Critic(BaseCritic):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def estimate_return(self, state: StateType) -> ValueType:
        return self.model(state)

    def update_fn(self, loss: type[float] | torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad(True)
        loss.backward()
        self.optimizer.step()
