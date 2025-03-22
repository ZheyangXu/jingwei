import gymnasium as gym
import numpy as np
import torch

from jingwei.infra.wrapper.data_wrapper import DataWrapper


def test_discrete_action_space():
    env = gym.make("CartPole-v1")
    data_wrapper = DataWrapper(env)
    observation, info = env.reset()
    action = env.action_space.sample()
    assert data_wrapper.wrap_action(action) == torch.tensor(action)
    assert data_wrapper.unwrap_action(torch.tensor(action)) == action
    assert data_wrapper.get_action_shape() == 2
    assert data_wrapper.get_observation_shape() == (4,)
    assert data_wrapper.device() == "cpu"
    assert data_wrapper.dtype() is None
    assert not data_wrapper.is_vec_env()
    assert data_wrapper.num_vec_env() == 1
    assert data_wrapper.unwrap_observation(observation=torch.tensor(observation)).shape == (4,)
    assert data_wrapper.wrap_observation(observation).shape == torch.tensor(observation).shape
    assert data_wrapper.to_numpy(torch.tensor([1, 2, 3]), np.float32).dtype == np.float32
    assert data_wrapper.to_tensor(np.array([1, 2, 3]), torch.float32, "cpu").dtype == torch.float32
    assert data_wrapper.to_tensor(np.array([1, 2, 3]), torch.float32, "cpu").device == torch.device(
        "cpu"
    )
