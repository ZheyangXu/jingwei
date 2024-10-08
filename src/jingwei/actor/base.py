from jingwei.infra.typing import ActionTensor, ObservationTensor, TensorType
import torch
import torch.nn as nn
import torch.optim as optim

from jingwei.domain.actor import BaseActor
from jingwei.domain.distribution import Distribution
from jingwei.infra.typing import *
from traitlets import observe


class Actor(BaseActor):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        distribution: Distribution,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.distribution = distribution
        self.device = device

    def get_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        logits = self.model(observation)
        self.distribution.prob_distribution(logits)
        return self.distribution.get_action(deterministic)
    
    def get_values(self, observation: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(observation)

    def get_probs(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def get_log_probs(self, observation: torch.Tensor) -> torch.Tensor:
        logits = self.model(observation)
        self.distribution.prob_distribution(logits)
        return self.distribution.log_prob(logits)

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def to(self, device: torch.device = None) -> torch.device:
        if device is None:
            device = self.device
        self.model.to(device)
        return device
