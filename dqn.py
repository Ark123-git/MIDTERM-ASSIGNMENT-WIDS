# import torch
# from torch import nn
# import torch.nn.functional as F

# class DQN(nn.Module):

#     def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
#         super(DQN, self).__init__()

#         self.enable_dueling_dqn=enable_dueling_dqn

#         self.fc1 = nn.Linear(state_dim, hidden_dim)

#         if self.enable_dueling_dqn:
            
#             self.fc_value = nn.Linear(hidden_dim, 256)
#             self.value = nn.Linear(256, 1)

          
#             self.fc_advantages = nn.Linear(hidden_dim, 256)
#             self.advantages = nn.Linear(256, action_dim)

#         else:
#             self.output = nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))

#         if self.enable_dueling_dqn:
         
#             v = F.relu(self.fc_value(x))
#             V = self.value(v)

        
#             a = F.relu(self.fc_advantages(x))
#             A = self.advantages(a)

#             # Calc Q
#             Q = V + A - torch.mean(A, dim=1, keepdim=True)

#         else:
#             Q = self.output(x)

#         return Q


# if __name__ == '__main__':
#     state_dim = 12
#     action_dim = 2
#     net = DQN(state_dim, action_dim)
#     state = torch.randn(10, state_dim)
#     output = net(state)
#     print(output)


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


# if __name__ == '__main__':
#     obs_dim = 12
#     act_dim = 2

#     net = DQN(obs_dim, act_dim)
#     sample_input = torch.randn(10, obs_dim)
#     result = net(sample_input)
#     print(result)
