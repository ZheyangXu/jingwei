from dataclasses import dataclass
from typing import List

import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class Batch(object):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    observation_next: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

    def observation_dtype(self) -> torch.dtype:
        return self.observation.dtype

    def action_dtype(self) -> torch.dtype:
        return self.action.dtype

    def __len__(self) -> int:
        return len(self.reward)

    def device(self) -> torch.device:
        return self.observation.device


class RolloutBatch(Batch):
    observation: NDArray | torch.Tensor
    action: NDArray | torch.Tensor
    reward: NDArray | torch.Tensor
    observation_next: NDArray | torch.Tensor
    terminated: NDArray | torch.Tensor
    truncated: NDArray | torch.Tensor
    episode_index: NDArray | torch.Tensor
    episode_end_position: List[int]

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
class AdvantagesWithReturnsBatch(Batch):
    log_prob: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
