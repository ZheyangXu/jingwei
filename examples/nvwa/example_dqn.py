import gymnasium as gym
import torch

from nvwa.agent.dqn import DQN
from nvwa.infra.module import QNet, QValueActionNet
from nvwa.trainer import OffPolicyTrainer


def main():
    env = gym.make("CartPole-v1")
    q_net = QNet(env.observation_space.shape[0], 128, env.action_space.n)

    dqn = DQN(q_net, env.action_space, env.observation_space)

    trainer = OffPolicyTrainer(
        dqn,
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

    double_q_net = QNet(env.observation_space.shape[0], 128, env.action_space.n)
    double_dqn = DQN(double_q_net, env.action_space, env.observation_space, is_double_dqn=True)
    trainer = OffPolicyTrainer(
        double_dqn,
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

    q_value_action_net = QValueActionNet(env.observation_space.shape[0], 128, env.action_space.n)
    dueling_dqn = DQN(q_value_action_net, env.action_space, env.observation_space)
    trainer = OffPolicyTrainer(
        dueling_dqn,
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
