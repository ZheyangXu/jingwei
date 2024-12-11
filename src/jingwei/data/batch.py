from dataclasses import dataclass

from jingwei.infra.typing import TensorLike


@dataclass(frozen=True)
class Batch(object):
    observation: TensorLike
    action: TensorLike
    observation_next: TensorLike
    terminated: TensorLike
    truncated: TensorLike
    done: TensorLike


@dataclass(frozen=True)
class LogProbBatch(Batch):
    values: TensorLike
    log_prob: TensorLike
