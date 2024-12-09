from typing import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from numpy.typing import NDArray


ShapeType = TypeVar("ShapeType", bound=tuple[int, ...])
DeviceType = TypeVar("DeviceType", bound=str | Any)
Dtype = TypeVar("Dtype", bound=Any)


@runtime_checkable
class TensorLike(Protocol, Generic[ShapeType, Dtype]):

    def __len__(self) -> int: ...

    def __add__(self, other: Any) -> Self: ...

    def __iadd__(self, other: Any) -> Self: ...

    def __mul__(self, other: Any) -> Self: ...

    def __imul__(self, other: Any) -> Self: ...

    @property
    def shape(self) -> ShapeType: ...

    @property
    def dtype(self) -> Dtype: ...

    def reshape(self, shap: ShapeType) -> Self: ...


type ObservationType = NDArray
type ActionType = NDArray | int
type RewardType = float
type DoneType = bool
