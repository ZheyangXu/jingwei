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

    def observation_dtype(self) -> torch.dtype | np.dtype:
        return self.observation.dtype

    def action_dtype(self) -> torch.dtype | np.dtype:
        return self.action.dtype

    def __len__(self) -> int:
        return len(self.reward)

    def device(self) -> torch.device | None:
        return self.observation.device if isinstance(self.observation, torch.Tensor) else None


class RolloutBatch(Batch):
    episode_index: NDArray
    _episode_end_position: List[int]

    def keys(self) -> List[str]:
        return [key for key in self.__dict__.keys() if not key.startswith("_")]

    def get_keys(self) -> List[str]:
        return [
            "observation",
            "action",
            "reward",
            "observation_next",
            "terminated",
            "truncated",
            "episode_index",
        ]


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
