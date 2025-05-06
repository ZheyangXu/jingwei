import gymnasium as gym
import numpy as np

from nvwa.data.transition import RolloutTransition, Transition


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
        truncated=truncated,
    )
    assert isinstance(transition.observation, np.ndarray)
    assert isinstance(transition.action, np.int64)
    assert isinstance(transition.reward, float)
    assert isinstance(transition.observation_next, np.ndarray)
    assert isinstance(transition.terminated, bool)
    assert isinstance(transition.truncated, bool)


def test_rollout_transition() -> None:
    env = gym.make("CartPole-v1")
    observation, info = env.reset()
    action = env.action_space.sample()
    observation_next, reward, terminated, truncated, info = env.step(action)
    log_prob = np.random.rand()
    values = np.random.rand()
    prob = np.random.rand()
    rollout_transition = RolloutTransition(
        observation=observation,
        action=action,
        reward=reward,
        observation_next=observation_next,
        terminated=terminated,
        truncated=truncated,
        log_prob=log_prob,
        values=values,
        prob=prob,
    )
    assert isinstance(rollout_transition.observation, np.ndarray)
    assert isinstance(rollout_transition.action, np.int64)
    assert isinstance(rollout_transition.reward, float)
    assert isinstance(rollout_transition.observation_next, np.ndarray)
    assert isinstance(rollout_transition.terminated, bool)
    assert isinstance(rollout_transition.truncated, bool)
    assert isinstance(rollout_transition.log_prob, float)
    assert isinstance(rollout_transition.values, float)
    assert isinstance(rollout_transition.prob, float)
    assert isinstance(rollout_transition.advantages, float)
