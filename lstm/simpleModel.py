from lstm import LSTM
import torch
import torch.nn as nn
from typing import List, Tuple

class SimpleModel1(nn.Module):
    def __init__(self, n_seq: int, seq_len:int, hidden_dim: int):   #（128，2，16）
        super(SimpleModel1, self).__init__()
        self.lstm=LSTM(input_dim=seq_len, hidden_dim=hidden_dim)
        self.fc = nn.Linear(n_seq * hidden_dim, 2)

    def forward(self,x: torch.Tensor,init_states: Tuple[torch.Tensor, torch.Tensor],):
        # x: (batch_size, n_seq, seq_len)
        # init_states：
        #   h_0: (batch_size, hidden_dim)
        #   c_0: (batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (n_seq, batch_size, seq_len)
        x, _ = self.lstm(x, init_states)
        # x: (n_seq, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (batch_size, n_seq, hidden_dim)
        x = x.reshape(x.size(0), -1)    # 将张量 x 的形状重新调整为 (batch_size, new_dim)
        # x: (batch_size, n_seq * hidden_dim)
        x = self.fc(x)
        # x: (batch_size, 2)
        return x
    
class SimpleModel4(nn.Module):
    def __init__(self, n_seq: int, seq_len:int, hidden_dim: int):   #（128，2，16）
        super(SimpleModel4, self).__init__()
        self.lstm=LSTM(input_dim=seq_len, hidden_dim=hidden_dim).to('cuda:0')
        #self.lstm=self.lstm.to('cuda:0')
        self.fc = nn.Linear(n_seq * hidden_dim, 2).to('cuda:1')
        #self.fc = self.fc.to('cuda:1')

    def forward(self,x: torch.Tensor,init_states: Tuple[torch.Tensor, torch.Tensor],):
        # x: (batch_size, n_seq, seq_len)
        # init_states：
        #   h_0: (batch_size, hidden_dim)
        #   c_0: (batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (n_seq, batch_size, seq_len)
        #x=x.to('cuda:0')
        init_states = (init_states[0].to('cuda:0'), init_states[1].to('cuda:0'))
        x, _ = self.lstm(x.to('cuda:0'), init_states)
        # x: (n_seq, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (batch_size, n_seq, hidden_dim)
        x = x.reshape(x.size(0), -1)    # 将张量 x 的形状重新调整为 (batch_size, new_dim)
        # x: (batch_size, n_seq * hidden_dim)
        #x=x.to('cuda:1')
        x = self.fc(x.to('cuda:1'))
        # x: (batch_size, 2)
        return x

class SimpleModel2(nn.Module):
    def __init__(self, n_seq: int, seq_len:int, hidden_dim: int):
        super(SimpleModel2, self).__init__()
        self.lstm=LSTM(input_dim=seq_len, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc2 = nn.Linear(int(hidden_dim / 2 * n_seq), 2)

    def forward(self,x: torch.Tensor,init_states: Tuple[torch.Tensor, torch.Tensor],):
        # x: (batch_size, n_seq, seq_len)
        # init_states：
        #   h_0: (batch_size, hidden_dim)
        #   c_0: (batch_size, hidden_dim)

        x = x.permute(1, 0, 2)
        # x: (n_seq, batch_size, seq_len)
        x, _ = self.lstm(x, init_states)
        # x: (n_seq, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (batch_size, n_seq, hidden_dim)

        x = self.fc1(x)
        # x: (batch_size, n_seq, hidden_dim / 2)
        x = x.reshape(x.size(0), -1)
        # x: (batch_size, n_seq * hidden_dim / 2)
        x = self.fc2(x)
        # x: (batch_size, 2)
        return x

class SimpleModel3(nn.Module):
    def __init__(self, n_seq: int, seq_len:int, hidden_dim: int):
        super(SimpleModel3, self).__init__()
        self.lstm1=LSTM(input_dim=seq_len, hidden_dim=hidden_dim)
        self.lstm2=LSTM(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc2 = nn.Linear(int(hidden_dim / 2 * n_seq), 2)

    def forward(self,x: torch.Tensor,init_states: Tuple[torch.Tensor, torch.Tensor],):
        # x: (batch_size, n_seq, seq_len)
        # init_states：
        #   h_0: (batch_size, hidden_dim)
        #   c_0: (batch_size, hidden_dim)

        x = x.permute(1, 0, 2)
        # x: (n_seq, batch_size, seq_len)
        x, states = self.lstm1(x, init_states)         # x: (n_seq, batch_size, hidden_dim)
        x, _ = self.lstm2(x, states)
        # x: (n_seq, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        # x: (batch_size, n_seq, hidden_dim)

        x = self.fc1(x)
        # x: (batch_size, n_seq, hidden_dim / 2)
        x = x.reshape(x.size(0), -1)
        # x: (batch_size, n_seq * hidden_dim / 2)
        x = self.fc2(x)
        # x: (batch_size, 2)
        return x
    

if __name__ == "__main__":
    batch_size = 32
    n_seq = 8 
    seq_len = 2
    hidden_dim = 16

    print("------------ SimpleModel1 --------------")
    x = torch.rand((batch_size, n_seq, seq_len))
    h_0 = torch.zeros((batch_size, hidden_dim))
    c_0 = torch.zeros((batch_size, hidden_dim))
    init_states = (h_0, c_0)
    model = SimpleModel1(n_seq, seq_len, hidden_dim)
    pred = model(x, init_states)
    print(pred.shape)  # (32, 2)

    print("------------ SimpleModel2 --------------")
    x = torch.rand((batch_size, n_seq, seq_len))
    h_0 = torch.zeros((batch_size, hidden_dim))
    c_0 = torch.zeros((batch_size, hidden_dim))
    init_states = (h_0, c_0)
    model = SimpleModel2(n_seq, seq_len, hidden_dim)
    pred = model(x, init_states)
    print(pred.shape)  # (32, 2)

    print("------------ SimpleModel3 --------------")
    x = torch.rand((batch_size, n_seq, seq_len))
    h_0 = torch.zeros((batch_size, hidden_dim))
    c_0 = torch.zeros((batch_size, hidden_dim))
    init_states = (h_0, c_0)
    model = SimpleModel2(n_seq, seq_len, hidden_dim)
    pred = model(x, init_states)
    print(pred.shape)  # (32, 2)