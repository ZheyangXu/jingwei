import gymnasium as gym
import torch
import torch.nn.functional as F

from nvwa.agent.sac import SAC
from nvwa.infra.module import QValueNetContinuous
from nvwa.trainer import OffPolicyTrainer


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int, action_bound: float):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(observation_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


def main() -> None:
    env = gym.make("Pendulum-v1")
    action_bound = env.action_space.high[0]
    policy_net = PolicyNetContinuous(
        env.observation_space.shape[0], 128, env.action_space.shape[0], action_bound
    )
    q_value_net = QValueNetContinuous(
        env.observation_space.shape[0], 128, env.action_space.shape[0]
    )

    sac = SAC(
        actor=policy_net,
        critic=q_value_net,
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=1e-3,
        sigma=0.2,
        tau=0.005,
        gamma=0.99,
    )

    trainer = OffPolicyTrainer(
        sac,
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
