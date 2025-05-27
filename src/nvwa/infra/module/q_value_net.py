import torch
import torch.nn as nn
import torch.nn.functional as F


class QValueNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observation, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
