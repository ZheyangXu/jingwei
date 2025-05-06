import gymnasium as gym
import numpy as np
import torch

from nvwa.infra.wrapper import DataWrapper


def test_data_wrapper() -> None:
    env = gym.make("CartPole-v1")
    wrapper = DataWrapper(env.observation_space, env.action_space)
    action = env.action_space.sample()

    wrappered_action = wrapper.wrap_action(action)
    unwrappered_action = wrapper.unwrap_action(wrappered_action)

    observation, _ = env.reset()

    wrappered_observation = wrapper.wrap_observation(observation)
    unwrappered_observation = wrapper.unwrap_observation(wrappered_observation)

    assert torch.equal(wrappered_action, torch.tensor(action))
    assert unwrappered_action == action
    assert wrappered_observation.shape == (4,)
    assert unwrappered_observation.shape == (4,)
    assert np.array_equal(unwrappered_observation, observation)
