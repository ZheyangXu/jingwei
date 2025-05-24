from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass()
class Batch(object):
    observation: torch.Tensor | NDArray
    action: torch.Tensor | NDArray
    reward: torch.Tensor | NDArray
    observation_next: torch.Tensor | NDArray
    terminated: torch.Tensor | NDArray
    truncated: torch.Tensor | NDArray
    episode_index: Optional[NDArray | List[int]] = None
    _episode_end_position: Optional[List[int]] = None
    old_log_prob: Optional[torch.Tensor | NDArray] = None
    returns: Optional[torch.Tensor | NDArray] = None
    advantage: Optional[torch.Tensor | NDArray] = None
    value: Optional[torch.Tensor | NDArray] = None
    dist: Optional[List[torch.distributions.Distribution] | torch.distributions.Distribution] = None

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

    def set_key(self, key: str, value: torch.Tensor | NDArray | List[Any]) -> None:
        if not hasattr(self, key) and value is None:
            raise ValueError(f"Key {key} does not exist in the batch.")
        setattr(self, key, value)
