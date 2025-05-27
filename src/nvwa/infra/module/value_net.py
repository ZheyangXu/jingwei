import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        return self.fc2(x)
