import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from jingwei.domain.critic.base import BaseCritic
from jingwei.infra.typing import *
from jingwei.infra.typing import LossType
from jingwei.transitions.base import TransitionBatch


class Critic(BaseCritic):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def estimate_return(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
