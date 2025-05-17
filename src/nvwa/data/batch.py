from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class Batch(object):
    observation: torch.Tensor | NDArray
    action: torch.Tensor | NDArray
    reward: torch.Tensor | NDArray
    observation_next: torch.Tensor | NDArray
    terminated: torch.Tensor | NDArray
    truncated: torch.Tensor | NDArray

    def get(self, key: str) -> torch.Tensor | NDArray:
        if key not in self.keys():
            raise ValueError(f"Key {key} does not exist in the batch.")
        return getattr(self, key)

    def observation_dtype(self) -> torch.dtype | np.dtype:
        return self.observation.dtype

    def action_dtype(self) -> torch.dtype | np.dtype:
        return self.action.dtype

    def __len__(self) -> int:
        return len(self.reward)

    def device(self) -> torch.device | None:
        return self.observation.device if isinstance(self.observation, torch.Tensor) else None

    def keys(self) -> List[str]:
        return [key for key in self.__dict__.keys() if not key.startswith("_")]

    def set_key(self, key: str, value: torch.Tensor | NDArray) -> None:
        if not hasattr(self, key) and value is None:
            raise ValueError(f"Key {key} does not exist in the batch.")
        setattr(self, key, value)


@dataclass(frozen=True)
class RolloutBatch(Batch):
    episode_index: NDArray
    _episode_end_position: List[int]


@dataclass(frozen=True)
class ReturnsBatch(Batch):
    returns: torch.Tensor | NDArray
    log_prob: torch.Tensor | NDArray


@dataclass(frozen=True)
class AdvantagesWithReturnsBatch(Batch):
    log_prob: torch.Tensor | NDArray
    values: torch.Tensor | NDArray
    advantages: torch.Tensor | NDArray
    returns: torch.Tensor | NDArray
