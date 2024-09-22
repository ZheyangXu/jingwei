import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from jingwei.actor.base import Actor
from jingwei.agent.dqn import DQNAgent
from jingwei.buffer.replay_buffer import ReplayBuffer
from jingwei.distributions.categorical import CategorialDistribution
from jingwei.infra.data_wrapper import DataWrapper
from jingwei.rollout.base import Rollout
from jingwei.trainner import Trainner


class QNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        return self.fc2(x)


class VAnet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc_a = nn.Linear(hidden_dim, action_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        a = self.fc_a(F.relu(self.fc1(observation)))
        v = self.fc_v(F.relu(self.fc1(observation)))
        return v + a - a.mean(1).view(-1, 1)


def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    observation_dim = env.observation_space.shape[0]
    hidden_dim = 10
    action_dim = env.action_space.n
    model = QNet(observation_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(model.parameters())
    device = torch.device("cpu")
    distribution = CategorialDistribution()
    actor = Actor(model, optimizer, distribution, device)
    agent = DQNAgent(actor)
    replay_buffer = ReplayBuffer(100000, env.observation_space.shape, action_dim)
    wrapper = DataWrapper(env.action_space, env.observation_space, torch.float32)
    rollout = Rollout(agent, env, wrapper)
    trainner = Trainner(agent, env, rollout, replay_buffer, wrapper, batch_size=2)
    trainner.train()


if __name__ == "__main__":
    main()
