import gymnasium as gym
import torch

from nvwa.agent.ppo import PPO
from nvwa.infra.module import PolicyContinuousNet, PolicyNet, ValueNet
from nvwa.trainer import OnPolicyTrainer


def main():
    env = gym.make("CartPole-v1")
    policy_net = PolicyNet(env.observation_space.shape[0], 128, env.action_space.n)
    value_net = ValueNet(env.observation_space.shape[0], 128)

    algo = PPO(policy_net, value_net, env.action_space, env.observation_space, 0.001)

    trainer = OnPolicyTrainer(
        algo,
        env,
        buffer_size=10000,
        max_epochs=5000,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    trainer.train()
    env.close()

    env = gym.make("Pendulum-v1")
    policy_net = PolicyContinuousNet(env.observation_space.shape[0], 128, env.action_space.shape[0])
    value_net = ValueNet(env.observation_space.shape[0], 128)
    algo = PPO(policy_net, value_net, env.action_space, env.observation_space, 0.001)
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
