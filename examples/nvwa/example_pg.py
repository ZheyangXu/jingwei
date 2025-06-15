import gymnasium as gym
import torch

from nvwa.agent.pg import PolicyGradient
from nvwa.infra.module import PolicyNet
from nvwa.trainer import OnPolicyTrainer


def main():
    env = gym.make("CartPole-v1")
    policy_net = PolicyNet(env.observation_space.shape[0], 128, env.action_space.n)

    pg = PolicyGradient(policy_net, env.action_space, env.observation_space)

    trainer = OnPolicyTrainer(
        pg,
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
