import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from nvwa.infra.wrapper import DataWrapper


def test_data_wrapper() -> None:
    env = gym.make("CartPole-v1")
    wrapper = DataWrapper(env.observation_space, env.action_space)

    observation, _ = env.reset()

    observation_tensor = wrapper.to_tensor(observation)

    tensor = torch.tensor(observation, dtype=wrapper.dtype, device=wrapper.device)
    unwrappered_observation = wrapper.to_numpy(tensor)

    assert observation_tensor.shape == wrapper.observation_shape
    assert observation_tensor.dtype == wrapper.dtype
    assert observation_tensor.device == wrapper.device
    assert unwrappered_observation.shape == wrapper.observation_shape
    assert unwrappered_observation.dtype == np.float32
