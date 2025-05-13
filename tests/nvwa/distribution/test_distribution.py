import gymnasium as gym
import numpy as np
import torch

from nvwa.distributions import CategoricalDistribution, GaussianDistribution


def test_categorical_distribution():
    action_space = gym.spaces.Discrete(5)
    distribution = CategoricalDistribution(action_space)
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    distribution.prob_distribution(logits)
    action = torch.tensor(1)

    assert distribution.action_dimension == 5
    assert distribution.log_prob(action).shape == torch.Size([])
    assert distribution.entropy().shape == torch.Size([])
    assert distribution.sample().shape == torch.Size([])
    assert distribution.mode().shape == torch.Size([])
    assert distribution.probs().shape == torch.Size([5])


def test_gaussian_distribution():
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    distribution = GaussianDistribution(action_space)
    mean = torch.tensor([[0.1, 0.2, 0.3]])
    std = torch.tensor([[0.1, 0.2, 0.3]])
    distribution.prob_distribution((mean, std))
    action = torch.tensor([[0.5, 0.5, 0.5]])

    assert distribution.action_dimension == 3
    assert distribution.log_prob(action).shape == torch.Size([])
    assert distribution.entropy().shape == torch.Size([])
    assert distribution.sample().shape == torch.Size([1, 3])
    assert distribution.mode().shape == torch.Size([1, 3])
    assert distribution.probs().shape == torch.Size([1, 3])
