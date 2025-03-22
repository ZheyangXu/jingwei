from typing import Any, Dict, List, Protocol, Self, runtime_checkable

from jingwei.infra.typing import TensorLike
from jingwei.protocol.data.batch import BatchProtocol


@runtime_checkable
class Actor(Protocol):

    def get_action(self, observation: TensorLike) -> TensorLike: ...

    def update_step(self, loss: TensorLike) -> None: ...

    def sample_action(
        self, observation: TensorLike, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> TensorLike: ...

    def set_train(self, mode: bool) -> bool: ...

    def set_eval(self, mode: bool) -> bool: ...


@runtime_checkable
class QActor(Actor, Protocol):

    def get_q_values(self, observation: TensorLike, action: TensorLike) -> TensorLike: ...

    def get_max_q_values(self, observation: TensorLike) -> TensorLike: ...

    def is_target_actor(self) -> bool: ...

    def update_target(self, actor: Self) -> None: ...

    def clone(self) -> Self: ...


@runtime_checkable
class PolicyActor(Actor, Protocol):

    def get_log_prob(self, observation: TensorLike) -> TensorLike: ...

    def get_prob(self, observation: TensorLike) -> TensorLike: ...

    def get_dist(self, batch: BatchProtocol) -> Any: ...
