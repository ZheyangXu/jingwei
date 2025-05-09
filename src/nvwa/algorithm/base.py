from abc import ABC, abstractmethod

import torch

from nvwa.data.batch import Batch


class Algorithm(ABC):

    @abstractmethod
    def get_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor: ...

    @abstractmethod
    def update(self, batch: Batch) -> None: ...

    @abstractmethod
    def to(self, device: torch.device) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...
