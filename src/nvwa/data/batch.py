from dataclasses import dataclass

import torch


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


@dataclass(frozen=True)
class RolloutBatch(Batch):
    log_prob: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
