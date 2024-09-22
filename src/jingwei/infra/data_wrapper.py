from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from jingwei.infra.typing import *
from jingwei.transitions.base import (
    TensorTransitionBatch,
    TransitionBatch,
    TransitionMembers,
)


class BaseDataWrapper(ABC):
    @abstractmethod
    def to_numpy(self, data: TensorType) -> np.ndarray:
        pass

    @abstractmethod
    def to_tensor(
        self,
        data: np.ndarray,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device | str] = "cpu",
    ) -> TensorType:
        pass

    @abstractmethod
    def unwrap_action(
        self,
        action: TensorType,
        preprocess_fn: Callable = lambda x: x,
        post_process_fn: Callable = lambda x: x,
    ) -> np.ndarray | int:
        pass

    @abstractmethod
    def observation_to_tensor(
        self,
        observation: np.ndarray,
        preprocess_fn: Callable = lambda x: x,
        post_process_fn: Callable = lambda x: x,
    ) -> TensorType:
        pass

    @abstractmethod
    def to_numpy_transition(
        self,
        transitions: TensorTransitionBatch,
        dtype: Optional[np.dtype] = None,
        preprocess_fn: Callable = lambda x: x,
        post_process_fn: Callable = lambda x: x,
    ) -> TransitionBatch:
        pass

    @abstractmethod
    def to_tensor_transition(
        self,
        transitions: TransitionBatch,
        dtype: Optional[torch.dtype] = None,
        preprocess_fn: Callable = lambda x: x,
        post_process_fn: Callable = lambda x: x,
    ) -> TensorTransitionBatch:
        pass


class DataWrapper(BaseDataWrapper):
    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        dtype: torch.dtype,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.dtype = dtype
        self.device = device
        self.num_envs = num_envs

    def to_numpy(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    def to_tensor(
        self,
        data: np.ndarray,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return torch.as_tensor(data, dtype=dtype, device=device)

    def unwrap_action(
        self,
        action: torch.Tensor,
        preprocess_fn: Callable[..., Any] = lambda x: x,
        post_process_fn: Callable[..., Any] = lambda x: x,
    ) -> np.ndarray | int:
        action = post_process_fn(self.to_numpy(preprocess_fn(action)))
        if isinstance(self.action_space, gym.spaces.Discrete) and action.ndim > 0:
            action = action[0]
        return action

    def observation_to_tensor(
        self,
        observation: np.ndarray,
        preprocess_fn: Callable[..., Any] = lambda x: x,
        post_process_fn: Callable[..., Any] = lambda x: x,
    ) -> torch.Tensor:
        if observation.ndim == len(self.observation_space.shape):
            observation = np.expand_dims(observation, 0)
        return post_process_fn(self.to_tensor(preprocess_fn(observation)))

    def to_numpy_transition(
        self,
        transitions: TensorTransitionBatch,
        dtype: Optional[np.dtype] = None,
        preprocess_fn: Callable[..., Any] = lambda x: x,
        post_process_fn: Callable[..., Any] = lambda x: x,
    ) -> TransitionBatch:
        preprocessed_transitions = preprocess_fn(transitions)
        batch = TransitionBatch(
            self.to_numpy(preprocessed_transitions.observation),
            self.to_numpy(preprocessed_transitions.action),
            self.to_numpy(preprocessed_transitions.reward),
            self.to_numpy(preprocessed_transitions.observation_next),
            self.to_numpy(preprocessed_transitions.terminated),
            self.to_numpy(preprocessed_transitions.truncated),
        )
        return post_process_fn(batch)

    def to_tensor_transition(
        self,
        transitions: TransitionBatch,
        dtype: Optional[torch.dtype] = None,
        preprocess_fn: Callable[..., TransitionBatch] = lambda x: x,
        post_process_fn: Callable[..., TensorTransitionBatch] = lambda x: x,
    ) -> TensorTransitionBatch:
        preprocessed_transitons = preprocess_fn(transitions)
        batch = TensorTransitionBatch(
            self.to_tensor(preprocessed_transitons.observation, dtype, self.device),
            self.to_tensor(preprocessed_transitons.action, dtype, self.device),
            self.to_tensor(preprocessed_transitons.reward, dtype, self.device),
            self.to_tensor(preprocessed_transitons.observation_next, dtype, self.device),
            self.to_tensor(preprocessed_transitons.terminated, dtype, self.device),
            self.to_tensor(preprocessed_transitons.truncated, dtype, self.device),
        )

        return post_process_fn(batch)
