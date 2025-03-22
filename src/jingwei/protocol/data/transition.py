from typing import Protocol, runtime_checkable

from jingwei.infra.typing import ActionType, DoneType, ObservationType


@runtime_checkable
class TransitionProtocol(Protocol):
    observation: ObservationType
    action: ActionType
    observation_next: ObservationType
    terminated: DoneType
    truncated: DoneType
