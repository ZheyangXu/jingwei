import torch
import torch.nn as nn
import torch.optim as optim


class Critic(object):
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

    def estimate_return(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def dtype(self) -> torch.dtype:
        self.mdoel.weight.dtype
