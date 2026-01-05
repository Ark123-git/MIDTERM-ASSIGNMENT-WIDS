import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_units=256, use_dueling=True):
        super(DQN, self).__init__()

        self.use_dueling = use_dueling

        self.linear1 = nn.Linear(obs_dim, hidden_units)

        if self.use_dueling:
            self.value_fc = nn.Linear(hidden_units, 256)
            self.value_head = nn.Linear(256, 1)

            self.adv_fc = nn.Linear(hidden_units, 256)
            self.adv_head = nn.Linear(256, act_dim)
        else:
            self.out = nn.Linear(hidden_units, act_dim)

    def forward(self, inp):
        feat = F.relu(self.linear1(inp))

        if self.use_dueling:
            v_feat = F.relu(self.value_fc(feat))
            v = self.value_head(v_feat)

            a_feat = F.relu(self.adv_fc(feat))
            a = self.adv_head(a_feat)

            q_vals = v + a - torch.mean(a, dim=1, keepdim=True)
        else:
            q_vals = self.out(feat)

        return q_vals



