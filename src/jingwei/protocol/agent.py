from typing import Protocol, runtime_checkable

from jingwei.infra.typing import TensorLike
from jingwei.protocol.data.batch import BatchProtocol


@runtime_checkable
class Agent(Protocol):

    def get_action(self, observation: TensorLike) -> TensorLike: ...

    def rtype(self) -> None: ...

    def is_on_policy(self) -> bool: ...

    def is_off_policy(self) -> bool: ...

    def is_offline(self) -> bool: ...

    def learn(self, batch: BatchProtocol) -> None: ...
