from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        return self.fc2(x)


class PolicyContinuousNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyContinuousNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
