from dataclasses import dataclass

from jingwei.infra.typing import ActionType, DoneType, ObservationType


@dataclass
class Transition(object):
    observation: ObservationType
    action: ActionType
    observation_next: ObservationType
    terminated: DoneType
    truncated: DoneType
