import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from nvwa.infra.functional import get_action_dimension, get_observation_shape


class DataWrapper(object):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.dtype = dtype
        self.device = device
        self.observation_shape = get_observation_shape(observation_space)
        self.action_dimension = get_action_dimension(action_space)
        self.observation_dim = len(self.observation_shape)

    def to_tensor(self, data: NDArray) -> torch.Tensor:
        return torch.tensor(data, dtype=self.dtype, device=self.device)

    def to_numpy(self, data: torch.Tensor) -> NDArray:
        return data.detach().cpu().numpy()

    def wrap_observation(self, observation: np.ndarray) -> torch.Tensor:
        if observation.ndim == self.observation_dim:
            observation = observation.reshape(1, -1)
        return self.to_tensor(observation)

    def unwrap_observation(self, observation: torch.Tensor) -> NDArray:
        if observation.ndim == self.observation_dim + 1:
            observation = observation.flatten().reshape(self.observation_shape)
        return self.to_numpy(observation)

    def wrap_action(self, action: NDArray | int) -> torch.Tensor:
        return self.to_tensor(action)

    def unwrap_action(self, action: torch.Tensor | int) -> NDArray | int:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return action.item()
        else:
            return np.squeeze(self.to_numpy(action), axis=0)
