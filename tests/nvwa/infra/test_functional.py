import gymnasium as gym
import numpy as np

from nvwa.infra.functional import get_action_dimension, get_observation_shape


def test_get_observation_shape() -> None:
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space
    observation_shape = get_observation_shape(observation_space)
    assert isinstance(observation_shape, tuple)
    assert observation_shape == (4,)


def test_get_action_dimension() -> None:
    env = gym.make("CartPole-v1")
    action_space = env.action_space
    action_dimension = get_action_dimension(action_space)
    assert isinstance(action_dimension, np.int64)
    assert action_dimension == 2
