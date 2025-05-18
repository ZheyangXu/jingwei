from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from matplotlib.pylab import f
from numpy.typing import NDArray

from nvwa.data.batch import Batch, RolloutBatch
from nvwa.infra.functional import get_action_dimension, get_action_type
from nvwa.infra.wrapper import DataWrapper


class Algorithm(nn.Module, ABC):

    def __init__(
        self,
        *,
        action_space: gym.Space,
        discount_factor: float = 0.99,
        observation_space: Optional[gym.Space] = None,
        is_action_scaling: bool = False,
        action_bound_method: Optional[str] = "clip",
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dimension = get_action_dimension(action_space)
        self.is_action_scaling = is_action_scaling
        self.action_bound_method = action_bound_method
        self.action_type = get_action_type(action_space)
        self.discount_factor = discount_factor
        self.dtype = dtype
        self.device = device
        self.wrapper = DataWrapper(observation_space, action_space, dtype, device)

    @abstractmethod
    def forward(
        self, observation: torch.Tensor, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def get_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor: ...

    def map_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_type == "continuous":
            with torch.no_grad():
                if self.action_bound_method == "clip":
                    action = torch.clamp(action, -1.0, 1.0)
                elif self.action_bound_method == "tanh":
                    action = torch.tanh(action)
                if self.is_action_scaling:
                    low, high = self.action_space.low, self.action_space.high
                    action = low + (high - low) * (action + 1.0) / 2.0
        return action

    def map_action_inverse(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_type == "continuous":
            with torch.no_grad():
                if self.is_action_scaling:
                    low, high = self.action_space.low, self.action_space.high
                    action = 2.0 * (action - low) / (high - low) - 1.0
                if self.action_bound_method == "tanh":
                    action = (torch.log(1.0 + action) - torch.log(1.0 - action)) / 2.0
        return action

    @abstractmethod
    def learn(self, batch: Batch) -> None: ...

    @abstractmethod
    def process_rollout(self, batch: RolloutBatch) -> Batch: ...

    def compute_episode_return(
        self,
        batch: RolloutBatch,
        values: Optional[NDArray] = None,
        values_next: Optional[NDArray] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[NDArray, NDArray]:
        if values_next is None:
            values_next = np.zeros_like(batch.reward, dtype=np.float32)
        values = np.roll(values_next, 1) if values is None else values

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        advantages = np.zeros_like(batch.reward)
        delta = batch.reward + gamma * values_next - values
        discount = (1.0 - end_flag) * (gamma * gae_lambda)
        advantage = 0.0
        for i in range(len(batch.reward) - 1, -1, -1):
            if i in batch._episode_end_position:
                advantage = 0.0
            advantage = delta[i] + discount[i] * advantage
            advantages[i] = advantage
        returns = advantages + values
        return returns, advantages

    def compute_nstep_return(
        self, batch: Batch, gamma: float = 0.99, n_step: int = 1, reward_norm: bool = False
    ) -> NDArray: ...

    def exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def process_func(self, batch: Batch, *args: Any, **kwargs: Any) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Meant to be overridden by subclasses. Typical usage is to add new keys to the
        batch, e.g., to add the value function of the next state. Used in :meth:`update`,
        which is usually called repeatedly during training.

        For modifying the replay buffer only once at the beginning
        (e.g., for offline learning) see :meth:`process_buffer`.
        """
        return batch
