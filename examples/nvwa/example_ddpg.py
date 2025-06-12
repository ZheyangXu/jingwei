import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.agent.ddpg import DDPG
from nvwa.infra.module import QValueNet
from nvwa.trainer import OffPolicyTrainer


class PolicyNet(nn.Module):
    def __init__(
        self, observation_dim: int, hidden_dim: int, action_dim: int, action_bound: float
    ) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        action = torch.tanh(self.fc2(x)) * self.action_bound
        return action


def main() -> None:
    env = gym.make("Pendulum-v1")
    action_bound = env.action_space.high[0]
    policy_net = PolicyNet(
        env.observation_space.shape[0], 128, env.action_space.shape[0], action_bound
    )
    q_value_net = QValueNet(env.observation_space.shape[0], 128, env.action_space.shape[0])

    ddpg = DDPG(
        actor=policy_net,
        critic=q_value_net,
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=1e-3,
        sigma=0.2,
        tau=0.005,
        gamma=0.99,
        estimate_step=1,
        is_action_scaling=True,
    )

    trainer = OffPolicyTrainer(
        ddpg,
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
    env.close()


if __name__ == "__main__":
    main()
