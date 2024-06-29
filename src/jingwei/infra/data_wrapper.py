from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch

from jingwei.infra.typing import *
from jingwei.transitions.base import TransitionBatch, TensorTransitionBatch, TransitionMembers


class BaseDataWrapper(ABC):
    @abstractmethod
    def to_numpy(self, data: TensorType) -> np.ndarray:
        pass

    @abstractmethod
    def to_tensor(self, data: np.ndarray, dtype: Any = None, device: str = None) -> TensorType:
        pass

    @abstractmethod
    def action_to_numpy(
        self, action: TensorType, preprocess_fn: Callable = lambda x: x, post_process_fn: Callable = lambda x: x
    ) -> np.ndarray | int:
        pass

    @abstractmethod
    def observation_to_tensor(
        self, observation: np.ndarray, preprocess_fn: Callable = lambda x: x, post_process_fn: Callable = lambda x: x
    ) -> TensorType:
        pass

    @abstractmethod
    def to_numpy_transition(
        self,
        transitions: TensorTransitionBatch,
        dtype: Any = None,
        preprocess_fn: Callable = lambda x: x,
        post_process_fn: Callable = lambda x: x,
    ) -> TransitionBatch:
        pass

    @abstractmethod
    def to_tensor_transition(
        self,
        transitions: TransitionBatch,
        dtype: Any = None,
        device: str = None,
        preprocess_fn: Callable = lambda x: x,
        post_process_fn: Callable = lambda x: x,
    ) -> TensorTransitionBatch:
        pass


class DataWrapper(BaseDataWrapper):
    def __init__(self, dtype: torch.dtype, device: torch.device = "cpu") -> None:
        self.dtype = dtype
        self.device = device

    def to_numpy(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    def to_tensor(self, data: np.ndarray, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return torch.as_tensor(data, dtype=dtype, device=device)

    def action_to_numpy(
        self,
        action: torch.Tensor,
        preprocess_fn: Callable[..., Any] = lambda x: x,
        post_process_fn: Callable[..., Any] = lambda x: x,
    ) -> np.ndarray | int:
        return post_process_fn(self.to_numpy(preprocess_fn(action)))

    def observation_to_tensor(
        self,
        observation: np.ndarray,
        preprocess_fn: Callable[..., Any] = lambda x: x,
        post_process_fn: Callable[..., Any] = lambda x: x,
    ) -> torch.Tensor:
        return post_process_fn(self.observation_to_tensor(preprocess_fn(observation)))

    def to_numpy_transition(
        self,
        transitions: TensorTransitionBatch,
        dtype: Any = None,
        preprocess_fn: Callable[..., Any] = lambda x: x,
        post_process_fn: Callable[..., Any] = lambda x: x,
    ) -> TransitionBatch:
        preprocessed_transitions = preprocess_fn(transitions)
        batch = TransitionBatch()
        for key in TransitionMembers.names:
            setattr(batch, key, self.to_numpy(getattr(preprocessed_transitions, key)))
        return post_process_fn(batch)

    def to_tensor_transition(
        self,
        transitions: TransitionBatch,
        dtype: Any = None,
        device: str = None,
        preprocess_fn: Callable[..., Any] = lambda x: x,
        post_process_fn: Callable[..., Any] = lambda x: x,
    ) -> TensorTransitionBatch:
        preprocessed_transitons = preprocess_fn(transitions)
        batch = TransitionBatch()
        for key in TransitionMembers.names:
            setattr(batch, key, self.to_tensor(getattr(preprocessed_transitons, key), dtype, device))
        return post_process_fn(batch)
