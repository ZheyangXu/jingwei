from typing import Protocol, runtime_checkable

from jingwei.infra.typing import TensorLike
from jingwei.protocol.data import BatchProtocol


@runtime_checkable
class Critic(Protocol):
    def estimate_return(self, batch: BatchProtocol) -> TensorLike: ...

    def update_step(self, loss: TensorLike) -> None: ...
