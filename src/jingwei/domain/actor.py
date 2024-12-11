from abc import ABC, abstractmethod
from typing import Any, Dict, List, Self

from jingwei.data.batch import Batch
from jingwei.infra.typing import TensorLike


class ActorBase(ABC):
    @abstractmethod
    def get_action(self, observation: TensorLike) -> TensorLike: ...

    @abstractmethod
    def update_step(self, loss: TensorLike) -> None: ...

    @abstractmethod
    def sample_action(
        self, observation: TensorLike, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> TensorLike: ...

    @abstractmethod
    def set_train(self, mode: bool) -> bool: ...

    @abstractmethod
    def set_eval(self, mode: bool) -> bool: ...


class QActorBase(ActorBase):
    @abstractmethod
    def get_q_values(self, observation: TensorLike, action: TensorLike) -> TensorLike: ...

    @abstractmethod
    def get_max_q_values(self, observation: TensorLike) -> TensorLike: ...

    @property
    @abstractmethod
    def is_target_actor(self) -> bool: ...

    @is_target_actor.setter
    @abstractmethod
    def is_target_actor(self, value: bool) -> None: ...

    @abstractmethod
    def update_target(self, actor: Self) -> None: ...

    @abstractmethod
    def clone(self) -> Self: ...


class PolicyActorBase(ActorBase):
    @abstractmethod
    def get_log_prob(self, observation: TensorLike) -> TensorLike: ...

    @abstractmethod
    def get_prob(self, observation: TensorLike) -> TensorLike: ...

    @abstractmethod
    def get_dist(self, batch: Batch) -> Any: ...
