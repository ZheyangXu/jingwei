from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvwa.agent.a2c import ActorCritic
from nvwa.data.batch import OldLogProbBatch, RolloutBatch
from nvwa.data.buffer import RolloutBuffer
from nvwa.distributions import Distribution


class PPO(ActorCritic):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        action_space: gym.Space,
        observation_space: Optional[gym.Space] = None,
        learning_rate: float = 0.001,
        distribution: Optional[Distribution] = None,
        n_epochs: int = 2,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.0,
        vf_coef: float = 0.5,
        normalize_advantages: bool = False,
        max_gradient_step: int = 1,
        batch_size: int = 256,
        lmbda: float = 0.9,
        eps: float = 0.2,
        max_grad_norm: float = 0.5,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        is_action_continuous: bool = False,
        is_action_scaling: bool = False,
        action_bound_method: Optional[str] = "clip",
    ) -> None:
        super().__init__(
            actor,
            critic,
            action_space,
            observation_space,
            learning_rate,
            distribution,
            discount_factor,
            gae_lambda,
            entropy_coef,
            vf_coef,
            max_gradient_step,
            batch_size,
            normalize_advantages,
            max_grad_norm,
            device,
            dtype,
            is_action_continuous,
            is_action_scaling,
            action_bound_method,
        )
        self.lmbda = lmbda
        self.eps = eps
        self.n_epochs = n_epochs

    def learn(self, buffer: RolloutBuffer) -> None:
        epoch_loss = {"actor_loss": 0, "critic_loss": 0, "loss": 0}
        for step in range(self.max_gradient_step):
            for batch in buffer.get_batch(self.batch_size):
                values, log_prob, entropy = self.evaluate_action(batch.observation, batch.action)
                advantage = batch.advantage
                if self.normalize_advantages:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                ratio = torch.exp(log_prob - batch.old_log_prob)
                policy_loss_1 = advantage * ratio

                policy_loss_2 = advantage * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                value_loss = F.mse_loss(batch.returns, values)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                epoch_loss["actor_loss"] += policy_loss.item()
                epoch_loss["critic_loss"] += value_loss.item()
                epoch_loss["loss"] += loss.item()

        return epoch_loss

    def process_rollout(self, batch: RolloutBatch) -> OldLogProbBatch:
        with torch.no_grad():
            observation = self.wrapper.wrap_observation(batch.observation)
            observation_next = self.wrapper.wrap_observation(batch.observation_next)
            action = self.get_action(observation)
            values = self.wrapper.to_numpy(self.compute_value(observation))
            values_next = self.wrapper.to_numpy(self.compute_value(observation_next))
            old_log_prob = self.wrapper.to_numpy(
                self.get_log_prob(
                    observation,
                    action,
                ).unsqueeze(dim=-1)
            )
        returns, advantage = self.compute_episode_return(
            batch,
            gamma=self.discount_factor,
            gae_lambda=self.gae_lambda,
            values=values,
            values_next=values_next,
        )

        return OldLogProbBatch(
            observation=batch.observation,
            action=batch.action,
            reward=batch.reward,
            observation_next=batch.observation_next,
            terminated=batch.terminated,
            truncated=batch.truncated,
            advantage=advantage,
            returns=returns,
            old_log_prob=old_log_prob,
        )
