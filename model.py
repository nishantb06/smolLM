from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class SmolLMConfig:
    block_size = 1024
    vocab_size = 49152
    n_layers = 30
    n_heads = 9
    n_embed = 576
    dropout = 0.1
    mlp_hidden_dim = 1536
    attention_dropout = 0.0
    dropout = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_heads
        self.n_embd = config.n_embed
        
        # Linear projections for Q, K, V
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # [n_embd, 3 * n_embd]
        self.c_proj = nn.Linear(config.n_embed, config.n_embed) # [n_embd, n_embd]
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # [B, T, n_embd]
        
        # Linear projection and split into Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # [B, T, n_embd] each
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd/n_head]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd/n_head]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd/n_head]
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5)) # [B, n_head, T, T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # [B, n_head, T, T]
        att = F.softmax(att, dim=-1) # [B, n_head, T, T]
        att = self.attn_dropout(att) # [B, n_head, T, T]
        
        # Weighted sum of values
        y = att @ v # [B, n_head, T, n_embd/n_head]
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B, T, n_embd]
        y = self.c_proj(y) # [B, T, n_embd]
        y = self.resid_dropout(y) # [B, T, n_embd]
        
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.silu    = nn.SilU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        return x