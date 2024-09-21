import torch
import torch.nn.functional as F

from jingwei.domain.actor import BaseActor
from jingwei.domain.agent import BaseAgent
from jingwei.domain.critic import BaseCritic
from jingwei.infra.mtype import MType
from jingwei.infra.typing import TensorType
from jingwei.transitions.base import TensorTransitionBatch


class PPOAgent(BaseAgent):
    def __init__(self, actor: BaseActor, critic: BaseCritic, epsilon: float = 0.1, gamma: float = 0.99) -> None:
        self.actor = actor
        self.critic = critic
        self.epsilon = epsilon
        self.gamma = gamma

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(observation)  # type: ignore

    def estimate_return(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        return self.critic.estimate_return(transitions=transitions)

    def update_step(self, transitions: TensorTransitionBatch) -> None:
        actor_loss = self.compute_actor_loss(transitions)
        critic_loss = self.compute_critic_loss(transitions)
        print(f"actor loss: {actor_loss}, critic loss: {critic_loss}")
        self.actor.update_step(actor_loss)
        self.critic.update_step(critic_loss)

    def compute_actor_loss(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        action_probs = self.actor.get_probs(transitions.observation_next)
        action_probs_old = self.actor.get_probs(transitions.observation)
        r_thelta = torch.div(action_probs, action_probs_old)
        clip = torch.clamp(r_thelta, min=1.0 - self.epsilon, max=1.0 + self.epsilon)
        advantage = self.advantage(transitions)
        loss = torch.sum(-torch.min(r_thelta * advantage, clip * advantage))
        entropy = torch.distributions.Categorical(action_probs.detach()).entropy()
        loss -= torch.sum(entropy)
        return loss
    
    def advantage(self, transitions: TensorTransitionBatch) -> float:
        return 1.

    def compute_critic_loss(self, transitions: TensorTransitionBatch) -> torch.Tensor:
        value = self.critic.estimate_return(transitions.observation_next)
        td_target = transitions.reward + self.gamma * value * (1 - transitions.terminated)
        return torch.mean(F.mse_loss(self.critic.estimate_return(transitions.observation), td_target))
    
    @property
    def mtype(self) -> MType:
        return MType.on_policy
