import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.algorithm.dqn import DQN
from nvwa.trainer import OffPolicyTrainer


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        return self.fc2(x)


def main():
    env = gym.make("CartPole-v1")
    q_net = QNet(env.observation_space.shape[0], 128, env.action_space.n)

    algo = DQN(q_net, env.observation_space, env.action_space)

    trainer = OffPolicyTrainer(
        algo,
        env,
        buffer_size=10000,
        minimal_size=320,
        max_epochs=2300,
        batch_size=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        gradient_step=10,
    )
    trainer.train()


if __name__ == "__main__":
    main()
