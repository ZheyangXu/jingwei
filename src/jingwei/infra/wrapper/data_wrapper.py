from typing import Any, Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from jingwei.infra.framework import FrameworkType
from jingwei.infra.functional.torch import to_numpy, to_tensor
from jingwei.infra.typing import DeviceType, Dtype, TensorLike


class DataWrapper(object):
    def __init__(
        self,
        env: gym.Env,
        dtype: Any = None,
        device: str = "cpu",
        framework: FrameworkType = FrameworkType.PYTORCH,
        to_numpy_func: Callable[[TensorLike, np.dtype], NDArray] = to_numpy,
        to_tensor_func: Callable[[NDArray, Dtype, DeviceType], TensorLike] = to_tensor,
    ) -> None:
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._dtype = dtype
        self._device = device
        self._is_vec_env: bool = False
        self._num_vec_env: int = 1
        self._framework = framework
        self._to_numpy_func = to_numpy_func
        self._to_tensor_func = to_tensor

    def observation_space(self) -> gym.Space:
        return self._observation_space

    def action_space(self) -> gym.Space:
        return self._action_space

    def get_observation_shape(self) -> tuple[int, ...]:
        return self._observation_space.shape

    def get_action_shape(self) -> tuple[int, ...] | int:
        if isinstance(self._action_space, gym.spaces.Discrete):
            return self._action_space.n
        return self._action_space.shape

    def device(self) -> str:
        return self._device

    def dtype(self) -> Any:
        return self._dtype

    def is_vec_env(self) -> bool:
        return self._is_vec_env

    def num_vec_env(self) -> int:
        return self._num_vec_env

    def unwrap_action(self, action: TensorLike) -> NDArray | int:
        if isinstance(self._action_space, gym.spaces.Discrete):
            return action.item()
        return self.to_numpy(action)

    def wrap_action(self, action: NDArray | int) -> TensorLike:
        if isinstance(self._action_space, gym.spaces.Discrete):
            return self.to_tensor(np.array([action]))
        return self.to_tensor(action)

    def unwrap_observation(self, observation: TensorLike) -> NDArray:
        return self.to_numpy(observation)

    def wrap_observation(self, observation: NDArray) -> TensorLike:
        return self.to_tensor(observation)

    def to_numpy(self, tensor: TensorLike, dtype: np.dtype = None) -> NDArray:
        return self._to_numpy_func(tensor, dtype)

    def to_tensor(self, arr: NDArray, dtype: Dtype = None, device: DeviceType = None) -> TensorLike:
        return self._to_tensor_func(arr, dtype, device)
