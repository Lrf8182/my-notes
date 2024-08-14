from lstm import LSTM
import torch
import torch.nn as nn
from typing import List, Tuple
from attention import MultiHeadAttentionV2

class ComplexModel(nn.Module):
    def __init__(self, n_seq: int, seq_len:int, hidden_dim: int,num_heads:int=4):
        super(ComplexModel, self).__init__()

        self.lstm1=LSTM(input_dim=seq_len, hidden_dim=hidden_dim).to('cuda:0')
        self.lstm2=LSTM(input_dim=hidden_dim, hidden_dim=hidden_dim).to('cuda:0')
        self.attention=MultiHeadAttentionV2(embed_dim=hidden_dim, num_heads=num_heads).to('cuda:0')
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim / 2)).to('cuda:1')
        self.fc2 = nn.Linear(int(hidden_dim / 2 * n_seq), 36)
        self.fc3=nn.Linear(36,6)
        self.fc4=nn.Linear(6,6)
        self.relu=nn.ReLU()
        self.seq1 = nn.Sequential(
            self.fc2,
            self.fc3,
            self.fc4
        ).to('cuda:0')
      

    def forward(self,x: torch.Tensor,init_states: Tuple[torch.Tensor, torch.Tensor],):
        # x: (batch_size, n_seq, seq_len)
        # init_states：
        #   h_0: (batch_size, hidden_dim)
        #   c_0: (batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (n_seq, batch_size, seq_len)
        init_states = (init_states[0].to('cuda:0'), init_states[1].to('cuda:0'))
        x, states = self.lstm1(x, init_states)


        x, _ = self.lstm2(x.to('cuda:0'), states)
        # x: (n_seq, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (batch_size, n_seq, hidden_dim)
        x = self.attention(x.to('cuda:0'))
        x = self.fc1(x.to('cuda:1'))
        # x: (batch_size, n_seq, hidden_dim / 2)
        x = x.reshape(x.size(0), -1)
        # x: (batch_size, n_seq * hidden_dim / 2)
        x = self.seq1(x.to('cuda:0'))
        x=self.relu(x)
        # x: (batch_size, 6)

        return x