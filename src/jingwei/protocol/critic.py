from typing import Protocol

from jingwei.data.batch import Batch
from jingwei.infra.typing import TensorLike


class Critic(Protocol):
    def estimate_return(self, batch: Batch) -> TensorLike: ...

    def update_step(self, loss: TensorLike) -> None: ...
