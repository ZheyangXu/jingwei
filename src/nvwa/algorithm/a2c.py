import torch
import torch.nn.functional as F

from nvwa.actor.actor import Actor
from nvwa.algorithm.base import OnPolicyAlgorithm
from nvwa.critic.base import Critic
from nvwa.data.batch import RolloutBatch


class ActorCritic(OnPolicyAlgorithm):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        gamma: float = 0.9,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

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
        td_target = batch.reward + self.gamma * self.critic.estimate_return(
            batch.observation_next
        ) * (1 - torch.logical_or(batch.terminated.long(), batch.truncated.long()).float())

        td_delta = td_target - self.critic.estimate_return(batch.observation_next)
        log_pros = torch.log(self.actor.model(batch.observation).gather(1, batch.action.long()))

        actor_loss = -torch.mean(log_pros * td_delta.detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic.estimate_return(batch.observation), td_target.detach())
        )
        self.actor.update_step(actor_loss)
        self.critic.update_step(critic_loss)
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
