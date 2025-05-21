import gymnasium as gym
import torch

from nvwa.agent.off_policy import OffPolicyAlgorithm
from nvwa.data.buffer import ReplayBuffer
from nvwa.data.transition import Transition
from nvwa.trainer.base import BaseTrainer


class OffPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        algo: OffPolicyAlgorithm,
        env: gym.Env,
        buffer_size: int = 10000,
        minimal_size: int = 320,
        max_epochs: int = 200,
        batch_size: int = 32,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gradient_step: int = 1,
        eval_episode_count: int = 10,
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
            eval_episode_count,
        )
        self.minimal_size = minimal_size

    def _init_buffer(self) -> None:
        self.buffer = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
        )

    def rollout(self) -> int:
        num_transitions = 0
        observation, _ = self.env.reset()
        while num_transitions < self.minimal_size:
            action = self.algo.get_behavior_action(self.wrapper.wrap_observation(observation))
            action = self.wrapper.unwrap_action(action)
            observation_next, reward, terminated, truncated, _ = self.env.step(action)
            transition = Transition(
                observation, action, reward, observation_next, terminated, truncated
            )
            self.buffer.put(transition)
            observation = observation_next
            num_transitions += 1
            if terminated or truncated:
                observation, _ = self.env.reset()
        return num_transitions

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            self.rollout()
            epoch_loss = 0.0
            for step in range(self.gradient_step):
                batch = self.buffer.sample(self.batch_size)
                status = self.algo.learn(batch)
                epoch_loss += status["loss"]

            if epoch % 10 == 0:
                eval_reward = self.evaluate()
                print(
                    f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {epoch_loss / self.gradient_step:.4f}, Evaluation Reward: {eval_reward:.4f}"
                )
