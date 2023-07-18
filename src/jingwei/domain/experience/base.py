
from dataclasses import dataclass

import numpy as np

from jingwei.infra.typing import *


@dataclass
class Experience(object):
    observation: ObservationType
    action: ActionType
    terminated: bool
    truncated: bool
    reward: RewardType
    next_observation: ObservationType


class Experiences(object):
    def __init__(self, capacity: int, batch_size: int) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer: List[Experience] = []
    
    @property
    def observations(self) -> ObservationsType:
        return np.array([experience.observation for experience in self.buffer])

    @property
    def actions(self) -> List[ActionType]:
        return np.array([experience.action for experience in self.buffer])

    @property
    def dones(self) -> List[bool]:
        pass

    @property
    def rewards(self) -> List[RewardType]:
        pass

    @property
    def next_observations(self) -> ObservationsType:
        pass

    @property
    def size(self) -> int:
        return len(self.buffer)

    def append(self, experience: Union[Experience, Tuple]) -> int:
        if isinstance(experience, Experience):
            self.buffer.append(experience)
        elif isinstance(experience, Tuple) and len(experience) == 4:
            self.buffer.append(Experience(observation=experience[0]))
        elif isinstance(experience)
        return len(self.buffer)