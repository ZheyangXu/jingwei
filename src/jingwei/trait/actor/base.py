from jingwei.domain.actor.base import BaseActor
from jingwei.domain.distributions.base import Distribution
from jingwei.infra.typing import *


class ActorTrait(BaseActor):
    def __init__(
        self, model: ModelType, optimizer: OptimizerType, distribution: Distribution, device: DeviceType
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.distribution = distribution
        self.device = device

    def get_action(self, observation: ObservationTensor, deterministic: bool = False) -> ActionType:
        logits = self.model(observation)
        self.distribution.prob_distribution(logits)
        return self.distribution.get_action(deterministic)

    def get_probs(self, observation: ObservationTensor) -> TensorType:
        return self.model(observation)

    def get_log_probs(self, observation: ObservationTensor) -> TensorType:
        action = self.model(observation)
        self.distribution.prob_distribution(action)
        return self.distribution.log_prob(action)

    def update_step(self, loss: LossType) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def to(self, device: DeviceType = None) -> DeviceType:
        if device is None:
            device = self.device
        self.model.to(device)
        return device
