import torch
import torch.nn as nn
import torch.nn.functional as F


class DDQN(nn.Module):
    def __init__(self, action_size=2, state_size=4, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        self.action_size = action_size
        self.state_size = state_size

        # simple 3 fully connected
        self.d1 = nn.Linear(state_size, 24)
        self.d2 = nn.Linear(24, 24)
        self.d3 = nn.Linear(24, action_size)  # outputs a tensor of q values

        self._initialize_weights()

    def _initialize_weights(self):
        """xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # relu for hidden + linear for output
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return self.d3(x)
