from abc import ABC, abstractmethod

import gymnasium as gym
from numpy.typing import DTypeLike, NDArray

from jingwei.infra.typing import (
    ActionType,
    DeviceType,
    Dtype,
    ObservationType,
    TensorLike,
)


class DataWrapperBase(ABC):
    @property
    @abstractmethod
    def observation_space(self) -> gym.Space: ...

    @property
    @abstractmethod
    def action_space(self) -> gym.Space: ...

    @property
    @abstractmethod
    def get_observation_shape(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def get_action_shape(self) -> tuple[int, ...] | int: ...

    @property
    @abstractmethod
    def device(self) -> DeviceType: ...

    @property
    @abstractmethod
    def dtype(self) -> Dtype: ...

    @property
    @abstractmethod
    def is_vec_env(self) -> bool: ...

    @property
    @abstractmethod
    def num_vec_env(self) -> int: ...

    @abstractmethod
    def unwrap_action(self, action: TensorLike) -> ActionType: ...

    @abstractmethod
    def wrap_action(self, action: ActionType) -> TensorLike: ...

    @abstractmethod
    def unwrap_observation(self, observation: TensorLike) -> ObservationType: ...

    @abstractmethod
    def wrap_observation(self, observation: ObservationType) -> TensorLike: ...

    @abstractmethod
    def to_numpy(self, tensor: TensorLike, dtype: DTypeLike) -> NDArray: ...

    @abstractmethod
    def to_tensor(self, arr: NDArray, dtype: Dtype, device: DeviceType) -> TensorLike: ...
