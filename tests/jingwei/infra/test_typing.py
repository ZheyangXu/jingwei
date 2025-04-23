import gymnasium as gym
import numpy as np
import pytest
import torch

from jingwei.infra.typing import GymEnvLike, TensorLike


def test_torch_tensor_is_tensor_like():
    assert isinstance(torch.tensor([1, 2, 3]), TensorLike)


def test_numpy_array_is_tensor_like():
    assert isinstance(np.array([1, 2, 3]), TensorLike)


def test_list_is_not_tensor_like():
    assert not isinstance([1, 2, 3], TensorLike)


paddle_tensor = None
try:
    import paddle

    PADDLE_AVAILABLE = True
    paddle_tensor = paddle.to_tensor([1, 2, 3])
except ModuleNotFoundError:
    PADDLE_AVAILABLE = False


@pytest.mark.skipif(not PADDLE_AVAILABLE, reason="PaddlePaddle is not installed")
def test_paddle_tensor_is_tensor_like():
    assert isinstance(paddle_tensor, TensorLike)


def test_gym_env_is_gym_env_like():
    env = gym.make("CartPole-v1")
    assert isinstance(env, GymEnvLike)
