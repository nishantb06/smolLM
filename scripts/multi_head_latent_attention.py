import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


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
    def __init__(self, hidden_size, num_heads, compression_ratio=4):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = hidden_size // compression_ratio

        self.q_proj_d = nn.Linear(hidden_size, self.latent_dim, bias=False)
        self.kv_proj_d = nn.Linear(hidden_size, self.latent_dim, bias=False)

        self.q_proj_u = nn.Linear(self.latent_dim, hidden_size, bias=False)
        self.k_proj_u = nn.Linear(self.latent_dim, hidden_size, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, hidden_size, bias=False)

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

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

        attn_output = scaled_dot_product_attention(
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



def test_mhla():
    torch.manual_seed(0)

    model = MultiHeadLatentAttention(
        hidden_size=512,
        num_heads=8,
        compression_ratio=4
    )

    x = torch.randn(2, 16, 512)

    with torch.no_grad():
        y = model(x)

    print(x.shape, y.shape)



if __name__ == "__main__":
    test_mhla()

