from typing import Any, Dict, Optional, Protocol, Self, TypeVar, runtime_checkable

from numpy.typing import NDArray


ShapeType = TypeVar("ShapeType", bound=tuple[int, ...])
DeviceType = str | Any
Dtype = Optional[Any]


@runtime_checkable
class TensorLike(Protocol):
    def __len__(self) -> int: ...

    def __add__(self, other: Any) -> Self: ...

    @property
    def shape(self) -> ShapeType: ...

    @property
    def dtype(self) -> Dtype: ...

    def reshape(self, shap: ShapeType) -> Self: ...


type ObservationType = NDArray
type ActionType = NDArray | int
type RewardType = float
type DoneType = bool
type InfoType = Dict[str, Any]


@runtime_checkable
class GymEnvLike(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[ObservationType, InfoType]: ...

    def step(
        self, action: ActionType
    ) -> tuple[ObservationType, RewardType, DoneType, DoneType, InfoType]: ...
