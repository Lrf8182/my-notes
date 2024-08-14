import torch
from torch import nn

torch.nn.MultiheadAttention


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, k_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim   # 嵌入维度，通常指的是输入向量的特征维度
        self.k_dim = embed_dim if k_dim is None else k_dim      # 查询向量的维度
        # Weight matrix Query
        self.wq = nn.Parameter(torch.randn(embed_dim, k_dim))   # nn.Parameter 类型确保这个权重矩阵会被自动注册为模型的可训练参数。
        # Weight matrix Key
        self.wk = nn.Parameter(torch.randn(embed_dim, k_dim))
        # Weight matrix Value
        self.wv = nn.Parameter(torch.randn(embed_dim, k_dim))

    def forward(self, x):
        # x: (batch_size, seq_len, d_m)
        # d_k: 代表键向量（Key）和查询向量（Query）的维度。在自注意力机制中，d_k 通常与 embed_dim 或 k_dim 相同，具体取决于你的实现。
        q = torch.matmul(x, self.wq)  # (batch_size, seq_len, d_k)                                 
        k = torch.matmul(x, self.wk)  # (batch_size, seq_len, d_k)
        v = torch.matmul(x, self.wv)  # (batch_size, seq_len, d_k)
        # attn_score: (batch_size, seq_len, seq_len)
        #attn_score = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.k_dim, dtype=q.dtype))
        attn_score=(torch.matmul(q,k.transpose(1,2))) / (self.k_dim ** 0.5)
        # attn_weight: (batch_size, seq_len, seq_len)
        attn_weight = torch.softmax(attn_score, dim=-1)
        # out: (batch_size, seq_len, d_k)
        out = torch.matmul(attn_weight, v)    # (batch_size, seq_len, d_k)
        return out


class MultiHeadAttentionV1(nn.Module):
    def __init__(self, embed_dim, k_dim, num_heads):
        super(MultiHeadAttentionV1, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = embed_dim if k_dim is None else k_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [SelfAttention(embed_dim=embed_dim, k_dim=k_dim) for _ in range(num_heads)]  # 每个实例用于实现一个注意力头。
        )
        # Weight that multiplies the final output of the heads
        self.w = nn.Parameter(torch.randn(num_heads * k_dim, embed_dim)) 

    def forward(self, x):
        # x: (batch_size, seq_len, d_m)
        # head: (batch_size, seq_len, d_k)
        # heads: (batch_size, seq_len, num_heads * d_k)
        heads = torch.cat([head(x) for head in self.heads], dim=-1)    
        # out: (batch_size, seq_len, embed_dim)
        out = torch.matmul(heads, self.w)   
        return out


class MultiHeadAttentionV2(nn.Module):
    def __init__(self, embed_dim, num_heads, k_dim=None):
        super(MultiHeadAttentionV2, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_dim = embed_dim if k_dim is None else k_dim
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim should be divisible by n_heads"
        self.wq = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))  
        self.wk = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))
        self.wv = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))
        self.wout = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        q = torch.matmul(x, self.wq)  # (batch_size, seq_len, embed_dim)
        k = torch.matmul(x, self.wk)  # (batch_size, seq_len, embed_dim)
        v = torch.matmul(x, self.wv)  # (batch_size, seq_len, embed_dim)
        # q, k, v: (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scores: (batch_size, n_heads, seq_len, seq_len)   
        # K:(batch_size, self.num_heads, self.head_dim, seq_len)-> (batch_size, self.num_heads, seq_len, seq_len)
        #scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        #    torch.tensor(self.head_dim, dtype=x.dtype)
        #)
        scores= (torch.matmul(q,k.transpose(-2,-1))) / (self.head_dim ** 0.5)
        # attn_weights: (batch_size, n_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        # attn_output: (batch_size, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (batch_size, seq_len, n_heads, head_dim)
        attn_output = (
            attn_output.transpose(1, 2)  # (batch_size, seq_len, n_heads, head_dim)
            .contiguous()  # Make sure the memory is contiguous
            .view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        )
        # out: (batch_size, seq_len, embed_dim)
        out = torch.matmul(attn_output, self.wout)

        return out



