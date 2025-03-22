import gymnasium as gym
import torch

from jingwei.data.batch import Batch, LogProbBatch
from jingwei.data.transition import Transition
from jingwei.protocol.data import BatchProtocol, LogProBatchProtocol, TransitionProtocol


def test_batch_is_batch_protocol() -> None:
    batch = Batch(
        observation=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        action=torch.tensor([0, 1]),
        reward=torch.tensor([0.0, 1.0]),
        observation_next=torch.tensor([[4, 5, 6], [7, 8, 9]]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, True]),
        done=torch.tensor([False, True]),
    )
    assert isinstance(batch, BatchProtocol)


def test_transition_is_transition_protocol() -> None:
    env = gym.make("CartPole-v1")
    observation, info = env.reset()
    action = env.action_space.sample()
    observation_next, reward, terminated, truncated, info = env.step(action)
    transition = Transition(observation, action, reward, observation_next, terminated, truncated)
    assert isinstance(transition, TransitionProtocol)


def test_log_pro_batch_is_log_pro_batch_protocol() -> None:
    log_prob_batch = LogProbBatch(
        observation=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        action=torch.tensor([0, 1]),
        reward=torch.tensor([0.0, 1.0]),
        observation_next=torch.tensor([[4, 5, 6], [7, 8, 9]]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, True]),
        done=torch.tensor([False, True]),
        values=torch.tensor([0.0, 1.0]),
        log_prob=torch.tensor([0.0, 1.0]),
    )
    assert isinstance(log_prob_batch, LogProBatchProtocol)
