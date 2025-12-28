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
    mlha_compression_ratio = 4
    # MoE parameters
    num_experts = 8
    num_shared_experts = 1
    top_k = 2


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
        self.w_q = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.w_k = nn.Linear(config.n_embed, config.n_embed// config.n_key_value_heads, bias=False)
        self.w_v = nn.Linear(config.n_embed, config.n_embed// config.n_key_value_heads, bias=False)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False) # [n_embd, n_embd]

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

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, position_ids, theta=10000.0):
    """
    q, k: (bsz, n_heads, seq_len, head_dim)
    position_ids: (bsz, seq_len)
    """
    head_dim = q.shape[-1]
    device = q.device

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    freqs = torch.einsum("bs,d->bsd", position_ids, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[:, None, :, :]
    sin = emb.sin()[:, None, :, :]

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class MultiHeadLatentAttention(nn.Module):
    def __init__(self,config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_heads

        assert config.n_embed % config.n_heads == 0

        self.hidden_size = config.n_embed
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.latent_dim = self.hidden_size // config.mlha_compression_ratio

        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)

        self.q_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.k_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        bsz, seq_len, _ = x.shape

        q_latent = self.q_proj_d(x)
        kv_latent = self.kv_proj_d(x)

        q = self.q_proj_u(q_latent)
        k = self.k_proj_u(kv_latent)
        v = self.v_proj_u(kv_latent)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(
            seq_len, device=x.device
        ).unsqueeze(0).expand(bsz, -1)

        q, k = apply_rope(q, k, position_ids)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=True,
            dropout_p=0.0,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.hidden_size)
        )

        return self.o_proj(attn_output)

class MLP(nn.Module):

    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, config.mlp_hidden_dim, bias=False)
        self.silu    = nn.SiLU()
        self.c_proj  = nn.Linear(config.mlp_hidden_dim, config.n_embed, bias=False)
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
        self.w1 = nn.Linear(config.n_embed, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, self.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepSeekExperLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias = False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias = False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias = False)
        self.activation_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMOE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts = 8, num_shared_experts = 1, top_k = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_experts - num_shared_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        self.shared_experts = nn.ModuleList([
            DeepSeekExperLayer(hidden_size,intermediate_size)
            for _ in range(self.num_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            DeepSeekExperLayer(hidden_size,intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        self.router = nn.Linear(hidden_size, self.num_routed_experts, bias = False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        shared_output = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts
        
        routing_logits = self.router(x) + self.routing_bias
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim = -1)

        scores = scores/ scores.sum(dim = -1, keepdim = True)
        
        combined_output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = indices[...,k]
            expert_scores = scores[..., k: k+1]

            # process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    combined_output[mask] += expert_output * expert_scores[mask]
        
        final_output = shared_output + combined_output
        return final_output
    
    def update_bias_term(self, expert_load):
        target_load = 1.0/ self.num_routed_experts
        load_diff = expert_load - target_load
        
        # for dynamic update rate based on the magnitude of the load imbalance
        update_rate = 0.1 * torch.abs(load_diff)
        
        # the update rate is static which is 0.1, need to be chosen as an hyper parameter
        # self.routing_bias -= 0.1 * load_diff
        self.routing_bias -= update_rate * load_diff
        
        return


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, num_shared_experts, top_k) -> None:
        super().__init__()
        self.moe = DeepSeekMOE(
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            num_experts = num_experts,
            num_shared_experts = num_shared_experts,
            top_k = top_k,
        )

    def forward(self, x):
        return self.moe(x)


class DecoderBlockWithRMSNorm(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config  = config
        self.rms_1 = RMSNorm(self.config.n_embed, eps=self.config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)
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
        self.attn = MultiHeadLatentAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: SmolLMConfig) -> None:
        super().__init__()
        self.config = config
        self.self_attn = MultiHeadLatentAttention(config)
        self.input_layernorm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(
            hidden_size=config.n_embed,
            intermediate_size=config.mlp_hidden_dim,
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            top_k=config.top_k
        )
    
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + residual

        # Feedforward
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = x + residual

        return x

class SmolLM(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embed) # [vocab_size, n_embd]
        # self.wpe = nn.Embedding(config.block_size, config.n_embed) # [max_seq_len, n_embd]
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layers)])
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
        
        # pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        # pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embd)
        x = self.wte(idx) # token embeddings of shape (B, T, n_embd)
        # x = x + pos_emb
        
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
