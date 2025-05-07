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
        gamma: float = 0.99,
        lmbda: float = 0.9,
        eps: float = 0.1,
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
        latent = self.actor.get_latent(observation)
        dist = torch.distributions.Categorical(logits=latent)
        log_prob = dist.log_prob(action.squeeze(-1))
        value = self.critic.estimate_return(observation)
        return value, log_prob.view(-1, 1), dist.entropy()

    def update(self, batch: RolloutBatch) -> None:
        values, log_probs, entropy = self.evaluate_action(batch.observation, batch.action)
        ratio = torch.exp(log_probs - batch.log_prob)

        # advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        advantages = batch.advantages

        policy_loss_1 = ratio * advantages
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        # print(f"returns: {batch.returns}, values: {values}")
        value_loss = torch.mean(F.mse_loss(values, batch.returns))

        self.actor.update_step(policy_loss)
        self.critic.update_step(value_loss)
        return {"actor_loss": policy_loss.item(), "critic_loss": value_loss.item()}
