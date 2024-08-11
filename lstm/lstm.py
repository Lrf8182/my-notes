import torch
import torch.nn as nn
from typing import List, Tuple


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,   #（2，16）
        hidden_dim: int,
    ):
        """
        Implementation of original LSTM cell.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        hidden_dim : int
            Hidden dimension.
        """

        super(LSTM, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim

        # Input Gate
        self.W_i = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_i = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        # Forget Gate
        self.W_f = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_f = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        # Cell Gate
        self.W_c = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_c = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_c = nn.Parameter(torch.Tensor(hidden_dim))

        # Output Gate
        self.W_o = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_o = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        self.fc = nn.Linear(128*(self.hidden_size), 1)

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.orthogonal_(param.data)   
            else:
                nn.init.zeros_(param.data)

    def forward(
        self,
        x: torch.Tensor,       #（128,batchsize,2)
        init_states: Tuple[torch.Tensor, torch.Tensor],   # (16,16)
    ):
        # Shape of x: (n_seq, batch_size, input_dim)
        # Shape of h_t: (batch_size, hidden_dim)
        # Shape of c_t: (batch_size, hidden_dim)
        h_t, c_t = init_states
        outputs = []
        n_seq = x.size(0)    
        for t in range(n_seq):
            # Shape of x_t: (batch_size, input_dim)
            x_t = x[t, :, :]                      
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)   # (16,16)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            c_hat_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t.unsqueeze(0))  # 在指定的维度上为张量添加一个新的维度
        # Shape of outputs: (n_seq, batch_size, hidden_dim)
        outputs = torch.cat(outputs, dim=0)   # 每次添加的 h_t 仍然是一个单独的张量，存储在 outputs 列表中。
        # 要将这些张量组合成一个整体，需要使用 torch.cat 将这些独立的张量沿第 0 维拼接起来。
        return outputs, (h_t, c_t)

