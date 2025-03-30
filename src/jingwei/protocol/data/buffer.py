from typing import Protocol, runtime_checkable

from jingwei.protocol.data.batch import BatchProtocol
from jingwei.protocol.data.transition import TransitionProtocol


@runtime_checkable
class BufferProtocol(Protocol):
    def add(self, transition: TransitionProtocol) -> int: ...

    def is_full(self) -> bool: ...

    def clean(self) -> int: ...

    def sample(self, batch_size: int) -> BatchProtocol: ...

    def get(self, batch_size: int) -> BatchProtocol: ...

    def reset(self): ...


@runtime_checkable
class ReplayBuffer(BufferProtocol, Protocol): ...


@runtime_checkable
class RolloutBuffer(BufferProtocol, Protocol): ...
