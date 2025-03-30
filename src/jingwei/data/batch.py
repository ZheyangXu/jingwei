from dataclasses import dataclass
from typing import List

from jingwei.infra.typing import TensorLike


class KeyEnabledBatch(object):
    def keys(self) -> List[str]:
        return self.__dict__.keys()


@dataclass(frozen=True)
class Batch(KeyEnabledBatch):
    observation: TensorLike
    action: TensorLike
    reward: TensorLike
    observation_next: TensorLike
    terminated: TensorLike
    truncated: TensorLike
    done: TensorLike


@dataclass(frozen=True)
class LogProbBatch(Batch):
    values: TensorLike
    log_prob: TensorLike
