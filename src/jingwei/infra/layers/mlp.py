# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_heads: int,
        num_outputs: int,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_heads)
        self.fc2 = nn.Linear(num_heads, num_outputs)
        self.activation = activation

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(state)))
