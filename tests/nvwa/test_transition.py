import gymnasium as gym
import numpy as np

from nvwa.data.transition import Transition


def test_transition() -> None:
    env = gym.make("CartPole-v1")
    observation, info = env.reset()
    action = env.action_space.sample()
    observation_next, reward, terminated, truncated, info = env.step(action)
    transition = Transition(
        observation=observation,
        action=action,
        reward=reward,
        observation_next=observation_next,
        terminated=terminated,
        truncate=truncated,
    )
    assert isinstance(transition.observation, np.ndarray)
    assert isinstance(transition.action, np.int64)
    assert isinstance(transition.reward, float)
    assert isinstance(transition.observation_next, np.ndarray)
    assert isinstance(transition.terminated, bool)
    assert isinstance(transition.truncate, bool)
