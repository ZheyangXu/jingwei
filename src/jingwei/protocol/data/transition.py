from typing import Protocol

from jingwei.infra.typing import ActionType, DoneType, ObservationType


class Transition(Protocol):
    observation: ObservationType
    action: ActionType
    observation_next: ObservationType
    terminated: DoneType
    truncated: DoneType
