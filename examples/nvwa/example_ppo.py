import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.algorithm.ppo import PPO
from nvwa.trainer import OnPolicyTrainer


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


def main():
    env = gym.make("CartPole-v1")
    policy_net = PolicyNet(env.observation_space.shape[0], 128, env.action_space.n)
    value_net = ValueNet(env.observation_space.shape[0], 128)

    algo = PPO(policy_net, value_net, 0.001)

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


if __name__ == "__main__":
    main()
