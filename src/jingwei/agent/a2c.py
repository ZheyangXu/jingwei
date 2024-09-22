import torch
import torch.nn.functional as F

from jingwei.domain.actor import BaseActor
from jingwei.domain.agent import BaseAgent
from jingwei.domain.critic import BaseCritic
from jingwei.infra.mtype import MType
from jingwei.infra.typing import *
from jingwei.transitions.base import TensorTransitionBatch


class ActorCriticAgent(BaseAgent):
    def __init__(self, actor: BaseActor, critic: BaseCritic, gamma: float = 0.99) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(observation)

    def estimate_return(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        return self.critic.estimate_return(transitions)

    def update_step(self, transitions: TensorTransitionBatch) -> None:
        actor_loss = self.compute_actor_loss(transitions)
        critic_loss = self.compute_critic_loss(transitions)
        self.actor.update_step(actor_loss)
        self.critic.update_step(critic_loss)

    def compute_actor_loss(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        td_targets = transitions.reward.reshape((32, 1)) + self.gamma * self.critic.estimate_return(
            transitions.observation_next
        ) * (1 - transitions.terminated).reshape((32, 1))
        td_delta = td_targets - self.critic.estimate_return(transitions.observation_next)
        log_probs = torch.log(
            self.actor.model(transitions.observation).gather(1, transitions.action.long())
        )
        return torch.mean(-log_probs * td_delta.detach())

    def compute_critic_loss(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        values = self.critic.estimate_return(transitions.observation)
        td_targets = transitions.reward + self.gamma * self.critic.estimate_return(
            transitions.observation_next
        ) * (1 - transitions.terminated)
        return torch.mean(F.mse_loss(values, td_targets.detach()))

    @property
    def mtype(self) -> MType:
        return MType.on_policy
