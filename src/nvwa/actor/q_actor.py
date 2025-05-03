from copy import deepcopy
from typing import Optional, Self

import torch
import torch.nn as nn
import torch.optim as optim

from nvwa.actor.base import BaseActor


class QActor(BaseActor):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epsilon: float = 0.1,
        device: torch.device = torch.device("cpu"),
        is_target_actor: bool = False,
    ) -> None:
        super().__init__(model, optimizer, device)
        self.epsilon = epsilon
        self._is_target_actor = is_target_actor

    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def get_action(self, observation: torch.Tensor) -> torch.Tensor | int:
        action_dist = self.model(observation)
        return action_dist.argmax()

    def get_max_q_values(
        self, observation: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if action is not None:
            max_q_values = self.model(observation).gather(1, action.long())
        else:
            max_q_values = self.model(observation).max(1)[0]

        return max_q_values

    def update_target(self, actor: Self) -> None:
        self.model.load_state_dict(actor.model.state_dict())

    def soft_update_target(self, actor: Self, tau: float = 0.05) -> None:
        self.model.load_state_dict(
            self.model.state_dict() + tau * (actor.model.state_dict() - self.model.state_dict())
        )

    def clone(self) -> Self:
        return deepcopy(self)

    @property
    def is_target_actor(self) -> bool:
        return self._is_target_actor

    @is_target_actor.setter
    def is_target_actor(self, value: bool) -> None:
        self._is_target_actor = value
