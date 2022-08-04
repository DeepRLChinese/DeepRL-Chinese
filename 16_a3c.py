import ray
from torch import nn
import torch.nn.functional as F
import torch


class QNet(nn.Module):
    """QNet.
    Input: feature
    Output: num_act of values
    """

    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)  # check?
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, dim_state, dim_action, max_action=2.0):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, dim_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x))
        return x


@ray.remote()
class Worker:
    def __init__(self, dim_state, dim_action):
        self.policy = PolicyNet(dim_state, dim_action, max_action=2.0):
        pass 

worker_lst = [Worker.remote() for _ in range(4)]
