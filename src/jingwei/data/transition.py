from dataclasses import dataclass
from typing import List

from jingwei.infra.typing import ActionType, DoneType, ObservationType


class KeyEnabledTransition(object):
    def keys(self) -> List[str]:
        return self.__dict__.keys()


@dataclass
class Transition(KeyEnabledTransition):
    observation: ObservationType
    action: ActionType
    reward: float
    observation_next: ObservationType
    terminated: DoneType
    truncated: DoneType
