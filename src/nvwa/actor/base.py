import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class BaseActor(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.model(observation)
        return action

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def dtype(self) -> torch.dtype:
        return self.model.weight.dtype
