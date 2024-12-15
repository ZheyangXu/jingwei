from abc import ABC, abstractmethod

from jingwei.data import Batch
from jingwei.infra.typing import TensorLike


class AgentBase(ABC):
    @abstractmethod
    def get_action(self, observation: TensorLike) -> TensorLike: ...

    @property
    @abstractmethod
    def rtype(self) -> None: ...

    @property
    @abstractmethod
    def is_on_policy(self) -> bool: ...

    @property
    @abstractmethod
    def is_off_policy(self) -> bool: ...

    @property
    @abstractmethod
    def is_offline(self) -> bool: ...

    @abstractmethod
    def learn(self, batch: Batch) -> None: ...
