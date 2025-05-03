import torch
import torch.nn as nn
import torch.optim as optim

from nvwa.actor.base import BaseActor


class Actor(BaseActor):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device | str = torch.device("cpu"),
    ):
        super().__init__(model, optimizer, device)

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        latent = self.get_latent(observation)
        dist = torch.distributions.Categorical(logits=latent)
        action = dist.sample()
        return action

    def get_latent(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def get_log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        logits = self.model(observation)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        return log_prob
