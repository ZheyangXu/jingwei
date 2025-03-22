from typing import Protocol, runtime_checkable

from jingwei.infra.typing import TensorLike


@runtime_checkable
class BatchProtocol(Protocol):
    observation: TensorLike
    action: TensorLike
    observation_next: TensorLike
    terminated: TensorLike
    truncated: TensorLike
    done: TensorLike


@runtime_checkable
class LogProBatchProtocol(BatchProtocol, Protocol):
    values: TensorLike
    log_prob: TensorLike
