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
    n_key_value_heads = 3
    rms_norm_eps = 1e-5


## Function which enables K and V to have less heads than Q. 
## it repeats the K and V heads n_rep times
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, n_kv_heads, slen,head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs,n_kv_heads, slen, n_rep, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_heads
        self.n_embd = config.n_embed
        
        # Linear projections for Q, K, V
        # self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # [n_embd, 3 * n_embd]
        self.w_q = nn.Linear(config.n_embed, config.n_embed)
        self.w_k = nn.Linear(config.n_embed, config.n_embed// config.n_key_value_heads)
        self.w_v = nn.Linear(config.n_embed, config.n_embed// config.n_key_value_heads)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed) # [n_embd, n_embd]

        self.n_rep = self.config.n_heads // self.config.n_key_value_heads

        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # [B, T, n_embd]
        
        # Linear projection and split into Q, K, V
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # [B, T, n_embd] each
        q = self.w_q(x) # [B, T, 576]
        k = self.w_k(x) # [B, T, 192]
        v = self.w_v(x) # [B, T, 192]
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.config.n_key_value_heads, k.size(-1) // self.config.n_key_value_heads).transpose(1, 2) # [B, 3, T, 64]
        q = q.view(B, T, self.config.n_heads, q.size(-1) // self.config.n_heads).transpose(1, 2) # [B, 9, T, 64]
        v = v.view(B, T, self.config.n_key_value_heads, v.size(-1) // self.config.n_key_value_heads).transpose(1, 2) # [B, 3, T, 64]
        
        # repeat k and v for each head
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5)) # [B, n_head, T, T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # [B, n_head, T, T]
        att = F.softmax(att, dim=-1) # [B, n_head, T, T]
        
        # Weighted sum of values
        y = att @ v # [B, n_head, T, n_embd/n_head]
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B, T, n_embd]
        y = self.c_proj(y) # [B, T, n_embd]
        y = self.resid_dropout(y) # [B, T, n_embd]
        
        return y

class MLP(nn.Module):

    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, config.mlp_hidden_dim)
        self.silu    = nn.SiLU()
        self.c_proj  = nn.Linear(config.mlp_hidden_dim, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        return x

class LlamaMLP(nn.Module):

    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_dim = config.mlp_hidden_dim # 1536
        self.w1 = nn.Linear(config.n_embed, self.hidden_dim)
        self.w2 = nn.Linear(self.hidden_dim, config.n_embed)
        self.w3 = nn.Linear(config.n_embed, self.hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DecoderBlockWithRMSNorm(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config  = config
        self.rms_1 = RMSNorm(self.config.n_embed, eps=self.config.rms_norm_eps)
        self.attn = CausalMultiHeadAttention(config)
        self.rms_2 = RMSNorm(self.config.n_embed, eps=self.config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x

class DecoderBlockWithLayerNorm(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SmolLM(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embed) # [vocab_size, n_embd]
        self.wpe = nn.Embedding(config.block_size, config.n_embed) # [max_seq_len, n_embd]
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlockWithRMSNorm(config) for _ in range(config.n_layers)])
        self.rms_norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps) # [n_embd]
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # [n_embd, vocab_size]
        
        # weight sharing
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embd)
        x = self.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = x + pos_emb
        
        # forward the blocks of the transformer
        for block in self.blocks:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.rms_norm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

if __name__ == "__main__":
    config = SmolLMConfig()
    model = SmolLM(config)
    # print number of parameters in Millions
    print(f"Number of parameters with RMSNorm: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    # test the model with a random input
    x = torch.randint(0, config.vocab_size, (1, 1024))
    logits, loss = model(x)
    print(f"Model output shape: {logits.shape}")
