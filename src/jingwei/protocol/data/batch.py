from typing import Protocol

from jingwei.infra.typing import TensorLike


class Batch(Protocol):
    observation: TensorLike
    action: TensorLike
    observation_next: TensorLike
    terminated: TensorLike
    truncated: TensorLike
    done: TensorLike


class LogProBatch(Batch, Protocol):
    values: TensorLike
    log_prob: TensorLike
