import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from jingwei.actor.base import Actor
from jingwei.agent.a2c import ActorCriticAgent
from jingwei.critic.base import Critic
from jingwei.distributions.categorical import CategorialDistribution
from jingwei.buffer.replay_buffer import ReplayBuffer
from jingwei.infra.data_wrapper import DataWrapper
from jingwei.rollout.base import Rollout
from jingwei.trainner import Trainner


class PolicyNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        return self.fc2(x)


def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    observation_dim = env.observation_space.shape[0]  # type: ignore
    hidden_dim = 100
    action_dim = env.action_space.n  # type: ignore

    actor_model = PolicyNet(observation_dim, hidden_dim, action_dim)
    actor_optimizer = optim.Adam(actor_model.parameters())
    distribution = CategorialDistribution()
    actor = Actor(actor_model, actor_optimizer, distribution)
    critic_model = ValueNet(observation_dim, hidden_dim)
    critic_optimizer = optim.Adam(critic_model.parameters())
    critic = Critic(critic_model, critic_optimizer)
    agent = ActorCriticAgent(actor, critic)
    replay_buffer = ReplayBuffer(10000, env.observation_space.shape, env.action_space.n)
    wrapper = DataWrapper(env.action_space, env.observation_space, torch.float32)
    rollout = Rollout(agent, env, wrapper)
    trainner = Trainner(agent, env, rollout, replay_buffer, wrapper)
    trainner.train()


if __name__ == "__main__":
    main()
