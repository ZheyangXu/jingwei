from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.algorithm.npg import NPG
from nvwa.trainer import OnPolicyTrainer


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        return self.fc2(x)


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)


class PolicyContinuousNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyContinuousNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


def main():
    env = gym.make("CartPole-v1")
    policy_net = PolicyNet(env.observation_space.shape[0], 128, env.action_space.n)
    value_net = ValueNet(env.observation_space.shape[0], 128)

    algo = NPG(policy_net, value_net, env.action_space, env.observation_space, 0.001)

    trainer = OnPolicyTrainer(
        algo,
        env,
        buffer_size=10000,
        max_epochs=500,
        batch_size=100,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    trainer.train()

    env.close()

    env = gym.make("Pendulum-v1")
    policy_net = PolicyContinuousNet(env.observation_space.shape[0], 128, env.action_space.shape[0])
    value_net = ValueNet(env.observation_space.shape[0], 128)
    algo = NPG(policy_net, value_net, env.action_space, env.observation_space)
    trainer = OnPolicyTrainer(
        algo,
        env,
        buffer_size=10000,
        max_epochs=5000,
        batch_size=10000,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
