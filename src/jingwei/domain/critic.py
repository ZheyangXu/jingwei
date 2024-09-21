from abc import ABC, abstractmethod

from jingwei.infra.typing import *


class BaseCritic(ABC):
    @abstractmethod
    def estimate_return(self, observation: ObservationType) -> ValueType: ...

    @abstractmethod
    def update_step(self, loss: LossType) -> None: ...
