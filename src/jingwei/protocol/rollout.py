from typing import Protocol, runtime_checkable


@runtime_checkable
class RolloutProtocol(Protocol):
    def rollout(self) -> int: ...

    def reset(self) -> None: ...
