from abc import ABC, abstractmethod
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

from jingwei.infra.typing import DeviceType
from jingwei.infra.utils.env import get_space_shape
from jingwei.protocol.data import BatchProtocol


class Buffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: DeviceType = "cpu",
        num_envs: int = 1,
    ) -> None:
        super(Buffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_sapce = observation_space
        self.action_space = action_space
        self.observation_shape = get_space_shape(observation_space)
        self.action_shape = get_space_shape(action_space)
        self.device = device
        self.num_envs = num_envs

        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        """Get the number of transitions in the buffer."""
        if self.full:
            return self.buffer_size
        return self.pos

    @property
    def size(self) -> int:
        """Get the size of the buffer."""
        return self.__len__()

    def capacity(self) -> int:
        """Get the capacity of the buffer."""
        return self.buffer_size

    def add(self, *args: List[Any], **kwargs: Dict[str, Any]) -> int:
        """Add a transition to the buffer."""
        return self.size

    def extend(self, *args: List[Any], **kwargs: Dict[str, Any]) -> int:
        """Extend the buffer with a list of transitions."""
        return self.size

    @abstractmethod
    def reset(self) -> int:
        """Reset the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> BatchProtocol:
        """Sample a batch of transitions from the buffer."""
        pass

    @abstractmethod
    def _get_batch(self, batch_indexies: np.ndarray) -> BatchProtocol:
        """Get a batch of transitions from the buffer."""
        pass
