import torch
import torch.nn.functional as F

from nvwa.actor.actor import Actor
from nvwa.algorithm.base import OnPolicyAlgorithm
from nvwa.critic.base import Critic
from nvwa.data.batch import RolloutBatch


class PPO(OnPolicyAlgorithm):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        n_epochs: int = 2,
        gamma: float = 0.9,
        lmbda: float = 0.9,
        eps: float = 0.2,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device
        self.n_epochs = n_epochs

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(observation)

    def estimate_value(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.actor.get_action(observation)
        log_prob = self.actor.get_log_prob(observation, action)
        return action, self.critic.estimate_return(observation), log_prob

    def evaluate_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob = self.actor.get_log_prob(observation, action)
        value = self.critic.estimate_return(observation)
        return log_prob, value, action

    def compute_advantages(self, batch: RolloutBatch) -> torch.Tensor:
        advantages = torch.zeros_like(batch.reward)
        last_advantage = 0
        for t in reversed(range(len(batch.reward))):
            if t == len(batch.reward) - 1:
                last_advantage = 0
            else:
                if batch.terminated[t] or batch.truncated[t]:
                    last_advantage = 0
                delta = batch.reward[t] + self.gamma * batch.values[t + 1] - batch.values[t]
                last_advantage = delta + self.gamma * self.lmbda * last_advantage
            advantages[t] = last_advantage
        return advantages

    def update(self, batch: RolloutBatch) -> None:
        values, log_probs, entropy = self.evaluate_action(batch.observation, batch.action)
        ratio = torch.exp(log_probs - batch.log_prob)

        advantages = self.compute_advantages(batch)

        policy_loss_1 = ratio * advantages
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        value_loss = F.mse_loss(values, batch.values)

        self.actor.update_step(policy_loss)
        self.critic.update_step(value_loss)
