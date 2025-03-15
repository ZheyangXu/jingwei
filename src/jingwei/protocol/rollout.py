from typing import Protocol


class Rollout(Protocol):
    def rollout(self) -> int: ...

    def reset(self) -> None: ...
