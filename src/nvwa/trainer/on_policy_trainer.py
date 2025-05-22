from typing import Optional, Tuple

import gymnasium as gym
import torch

from nvwa.agent.base import BaseAgent
from nvwa.data.buffer import RolloutBuffer
from nvwa.trainer.base import BaseTrainer


class OnPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        algo: BaseAgent,
        env: gym.Env,
        buffer_size: int = 10000,
        max_epochs: int = 200,
        batch_size: int = 32,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gradient_step: int = 2,
        n_rollout_step: Optional[int] = None,
        n_episodes: int = 100,
    ) -> None:
        super().__init__(
            algo,
            env,
            buffer_size,
            max_epochs,
            batch_size,
            device,
            dtype,
            gradient_step,
            n_rollout_step,
            n_episodes,
        )
        self.n_episodes = n_episodes

    def _init_buffer(self) -> None:
        self.buffer = RolloutBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
        )

    def rollout(self) -> Tuple[int, float]:
        self.rolloutor.reset()
        rollout_batch = self.rolloutor.rollout()
        enriched_rollout_batch = self.algo.process_rollout(rollout_batch)
        self.buffer.reset()
        self.buffer.add(enriched_rollout_batch)

        return self.buffer.size(), rollout_batch.reward.sum() / self.n_episodes

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            num_transitions, reward = self.rollout()
            epoch_loss = self.algo.learn(self.buffer)

            if epoch % 10 == 0:
                eval_reward = self.evaluate()
                print(
                    f"Epoch {epoch}/{self.max_epochs}, Loss: {epoch_loss['loss'] / self.gradient_step:.4f}, Actor Loss: {epoch_loss['actor_loss'] / self.gradient_step:.4f}, Critic Loss: {epoch_loss['critic_loss'] / self.gradient_step:.4f}, Eval Reward: {eval_reward}, rollout Reward: {reward:.4f}, Transitions: {num_transitions}"
                )
