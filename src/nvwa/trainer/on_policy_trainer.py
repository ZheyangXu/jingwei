import gymnasium as gym
import torch

from nvwa.algorithm.base import OnPolicyAlgorithm
from nvwa.data.buffer import RolloutBuffer
from nvwa.data.transition import RolloutTransition
from nvwa.trainer.base import BaseTrainer


class OnPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        algo: OnPolicyAlgorithm,
        env: gym.Env,
        buffer_size: int = 10000,
        max_epochs: int = 200,
        batch_size: int = 32,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gradient_step: int = 1,
        eval_episode_count: int = 2,
        n_episodes: int = 10,
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
        self.n_episodes = n_episodes

    def _init_buffer(self) -> None:
        self.buffer = RolloutBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
        )

    def rollout(self) -> int:
        self.buffer.reset()
        num_transitions = 0
        for n_step in range(self.n_episodes):
            observation, _ = self.env.reset()
            done = False
            num_episodes = 0
            while not done:
                action, values, log_probs = self.algo.estimate_value(
                    self.wrapper.wrap_observation(observation)
                )
                action = self.wrapper.unwrap_action(action)
                observation_next, reward, terminated, truncated, _ = self.env.step(action)

                transition = RolloutTransition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    observation_next=observation_next,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=log_probs,
                    values=values,
                )
                num_transitions += self.buffer.put(transition)
                observation = observation_next
                done = terminated or truncated
                num_episodes += 1

            self.buffer.compute_advantage(num_episodes)

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            self.rollout()
            epoch_loss = 0.0
            for step in range(self.gradient_step):
                for batch in self.buffer.get_batch(self.batch_size):
                    status = self.algo.update(batch)
                    epoch_loss += status["loss"]

            if epoch % 10 == 0:
                eval_reward = self.evaluate()
                print(
                    f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {epoch_loss / self.gradient_step:.4f}, Eval Reward: {eval_reward}"
                )
