# import for  colab/kaggle
# !pip install datasets transformers -q
# !pip install pytorch-lightning lightning tiktoken -q
import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2Tokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint

# =====================================================
# Training Hyperparameters
# =====================================================
block_size = 512
batch_size = 8
max_lr = 1e-3
warmup_steps = 10
max_steps = 25000
log_every_n_steps = 100
save_checkpoints_every_n_steps = 10
effective_batch_size = 32

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
    "HuggingFaceTB/cosmo2-tokenizer"
)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size


# =====================================================
# Model Configuration
# =====================================================
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
    n_key_value_heads = 3
    rms_norm_eps = 1e-5
    mlha_compression_ratio = 4
    # MoE parameters
    num_experts = 8
    num_shared_experts = 1
    top_k = 2


# =====================================================
# Model Components
# =====================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Function which enables K and V to have less heads than Q.
    It repeats the K and V heads n_rep times.
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, n_kv_heads, slen, n_rep, head_dim)
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


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, position_ids, theta=10000.0):
    """
    Apply Rotary Position Embedding (RoPE) to queries and keys.
    
    Args:
        q, k: (bsz, n_heads, seq_len, head_dim)
        position_ids: (bsz, seq_len)
    """
    head_dim = q.shape[-1]
    device = q.device

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[:, None, :, :]
    sin = emb.sin()[:, None, :, :]

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLHA) - compresses Q, K, V through a latent bottleneck
    for more efficient attention computation.
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_heads

        assert config.n_embed % config.n_heads == 0

        self.hidden_size = config.n_embed
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.latent_dim = self.hidden_size // config.mlha_compression_ratio

        # Down-projection to latent space
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)

        # Up-projection from latent space
        self.q_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.k_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # Compress to latent space
        q_latent = self.q_proj_d(x)
        kv_latent = self.kv_proj_d(x)

        # Expand from latent space
        q = self.q_proj_u(q_latent)
        k = self.k_proj_u(kv_latent)
        v = self.v_proj_u(kv_latent)

        # Reshape for multi-head attention
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        position_ids = torch.arange(
            seq_len, device=x.device
        ).unsqueeze(0).expand(bsz, -1)
        q, k = apply_rope(q, k, position_ids)

        # Scaled dot-product attention (Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=True,
            dropout_p=0.0,
        )

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.hidden_size)
        )

        return self.o_proj(attn_output)


class DeepSeekExpertLayer(nn.Module):
    """Single expert in the MoE layer using SwiGLU activation."""
    def __init__(self, hidden_size, intermediate_size) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMOE(nn.Module):
    """
    DeepSeek-style Mixture of Experts with shared experts and routed experts.
    Uses sigmoid routing with bias-based load balancing.
    """
    def __init__(self, hidden_size, intermediate_size, num_experts=8, num_shared_experts=1, top_k=2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_experts - num_shared_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_shared_experts)
        ])
        
        # Routed experts (conditionally activated based on routing)
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        # Router and bias for load balancing
        self.router = nn.Linear(hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # Process through shared experts
        shared_output = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts

        # Compute routing probabilities
        routing_logits = self.router(x) + self.routing_bias
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Normalize scores
        scores = scores / scores.sum(dim=-1, keepdim=True)

        # Process through routed experts
        combined_output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = indices[..., k]
            expert_scores = scores[..., k:k+1]

            # Process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    combined_output[mask] += expert_output * expert_scores[mask]

        final_output = shared_output + combined_output
        return final_output

    def update_bias_term(self, expert_load):
        """
        Update routing bias to improve load balancing.
        Called after processing a batch during training.
        """
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load

        # Dynamic update rate based on the magnitude of load imbalance
        update_rate = 0.1 * torch.abs(load_diff)

        # Update bias to discourage overloaded experts and encourage underloaded ones
        self.routing_bias.data -= update_rate * load_diff


class LlamaMLP(nn.Module):
    """MLP layer using DeepSeek MoE."""
    def __init__(self, hidden_size, intermediate_size, num_experts, num_shared_experts, top_k) -> None:
        super().__init__()
        self.moe = DeepSeekMOE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
        )

    def forward(self, x):
        return self.moe(x)


class LlamaDecoderLayer(nn.Module):
    """Transformer decoder layer with MLHA and MoE."""
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
        # Self-attention with residual
        residual = x
        x = self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + residual

        # Feedforward (MoE) with residual
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = x + residual

        return x


class SmolLM(nn.Module):
    """
    SmolLM with Multi-Head Latent Attention (MLHA) and Mixture of Experts (MoE).
    Uses RoPE for positional encoding instead of learned position embeddings.
    """
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)  # [vocab_size, n_embd]
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layers)])
        self.rms_norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight sharing between token embeddings and output projection
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

        # Token embeddings (no position embeddings - using RoPE in attention)
        x = self.wte(idx)  # (B, T, n_embd)

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.rms_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# =====================================================
# Data Loading
# =====================================================

def load_cosmopedia_dataset(batch_size=8, seq_length=1024):
    """
    Returns a torch dataloader for the cosmopedia dataset
    """
    try:
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            name="cosmopedia-v2",
            split="train",
            streaming=True,
        )

        def encode(examples):
            tokens = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=seq_length + 1,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0).clone().detach()
            input_ids = torch.clamp(input_ids, min=0, max=tokenizer.vocab_size - 1)
            labels = input_ids.clone().detach()
            labels = labels[1:].to(torch.int64)
            input_ids = input_ids[:-1].to(torch.int64)

            return {"input_ids": input_ids, "labels": labels}

        dataset = dataset.map(encode, remove_columns=["text"], batched=False)
        dataset = dataset.with_format("torch")
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader
    except Exception as e:
        print(e)
        return None


# =====================================================
# Expert Load Balancing Utilities
# =====================================================

def compute_expert_load(model, input_ids, device):
    """
    Compute expert load based on routing decisions.
    This is called after forward pass to track how tokens are distributed across experts.
    """
    # Get the first layer's MoE module for reference
    first_moe = model.model.blocks[0].mlp.moe
    num_routed_experts = first_moe.num_routed_experts
    top_k = first_moe.top_k

    with torch.no_grad():
        # Get token embeddings (same as in model forward)
        hidden_states = model.model.wte(input_ids)

        # Compute expert load based on first layer's routing
        expert_load = torch.zeros(num_routed_experts, device=device)

        # Get routing decisions from the first layer
        # Apply input layernorm first (as in LlamaDecoderLayer)
        normed_hidden = model.model.blocks[0].input_layernorm(hidden_states)
        routing_logits = first_moe.router(normed_hidden) + first_moe.routing_bias
        routing_probs = torch.sigmoid(routing_logits)
        _, indices = torch.topk(routing_probs, top_k, dim=-1)

        for k in range(top_k):
            for i in range(num_routed_experts):
                expert_load[i] += (indices[..., k] == i).sum()

        # Normalize by total number of routing decisions
        expert_load = expert_load / (input_ids.size(0) * input_ids.size(1) * top_k)

    return expert_load


def update_all_moe_bias_terms(model, expert_load):
    """
    Update bias terms for all MoE layers in the model.
    This helps improve load balancing over time.
    """
    for block in model.model.blocks:
        block.mlp.moe.update_bias_term(expert_load)


# =====================================================
# PyTorch Lightning Module
# =====================================================

class SmolLMMoEMLHALightning(pl.LightningModule):
    def __init__(self, config: SmolLMConfig, lr, warmup_steps, max_steps):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = SmolLM(self.config)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.generation_prompt = "Once upon a time"
        self._generating = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        target_ids = batch["labels"]
        logits, _ = self(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Log the loss
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True
        )

        # =====================================================
        # Expert Load Balancing - Update MoE bias terms
        # This is done after forward pass to improve load balancing
        # =====================================================
        with torch.no_grad():
            expert_load = compute_expert_load(self, input_ids, self.device)
            update_all_moe_bias_terms(self, expert_load)

            # Log expert load distribution for monitoring
            for i, load in enumerate(expert_load):
                self.log(f"expert_load_{i}", load.item(), on_step=True, on_epoch=False, logger=True)

        # Generate text every n steps
        if (self.global_step) % log_every_n_steps == 0 and not self._generating:
            self._generating = True
            self.generate_and_log_sample()
            self._generating = False

        return loss

    def generate_and_log_sample(self):
        """Generate and log a sample of text from the model"""
        try:
            # Encode the prompt
            prompt_ids = self.tokenizer.encode(
                self.generation_prompt, return_tensors="pt"
            ).to(self.device)

            # Generate new tokens
            generated_ids = self.generate(
                prompt_ids, max_new_tokens=50, temperature=0.8, top_k=40
            )

            # Decode the generated tokens
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())

            # Create a formatted message
            message = (
                f"\n{'='*40}\n"
                f"Step {self.global_step} generation:\n"
                f"Prompt: {self.generation_prompt}\n"
                f"Generated: {generated_text}\n"
                f"{'='*40}\n"
            )

            print(message)
        except Exception as e:
            print(f"Generation failed with error: {str(e)}")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text given a starting sequence of tokens.

        Args:
            idx (torch.Tensor): Starting token indices, shape (B, T)
            max_new_tokens (int): Number of tokens to generate
            temperature (float): Sampling temperature
            top_k (int): If specified, only sample from the top k most probable tokens
        """
        for _ in range(max_new_tokens):
            # Crop if sequence is too long
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            # Forward pass
            logits, _ = self(idx_cond)
            # Get logits for last token and scale by temperature
            logits = logits[:, -1, :] / temperature
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return self.hparams.lr * (current_step + 1) / self.hparams.warmup_steps
            elif current_step > self.hparams.max_steps:
                return self.hparams.lr * 0.1
            decay_ratio = (current_step - self.hparams.warmup_steps) / (
                self.hparams.max_steps - self.hparams.warmup_steps
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.hparams.lr * 0.1 + coeff * (
                self.hparams.lr - self.hparams.lr * 0.1
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]


# =====================================================
# Main Training Script
# =====================================================

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    dataloader = load_cosmopedia_dataset(batch_size=batch_size, seq_length=block_size)

    # Check if checkpoint exists
    checkpoint_path = "checkpoints/moe_mlha_best-checkpoint.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = SmolLMMoEMLHALightning.load_from_checkpoint(
            checkpoint_path,
            config=SmolLMConfig(),
            lr=max_lr,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
        )
    else:
        print("Starting training from scratch")
        model = SmolLMMoEMLHALightning(SmolLMConfig(), max_lr, warmup_steps, max_steps)

    # Print model info
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Local CSV logger (logs saved to logs/ directory)
    os.makedirs("logs", exist_ok=True)
    csv_logger = CSVLogger(
        save_dir="logs",
        name="moe_mlha_experiment",
    )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="moe_mlha_best-checkpoint",
        verbose=True,
        every_n_train_steps=save_checkpoints_every_n_steps,
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    progress_bar = RichProgressBar(
        refresh_rate=1,
        leave=False,
        theme=RichProgressBarTheme(
            description="",
            progress_bar="#6206E0",
            progress_bar_finished="#6206E0",
            progress_bar_pulse="#6206E0",
            batch_progress="",
            time="dim",
            processing_speed="dim underline",
            metrics="italic",
            metrics_text_delimiter=" ",
            metrics_format=".3f",
        ),
        console_kwargs=None,
    )

    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator=device,
        devices=1,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            progress_bar,
            checkpoint_callback,
        ],
        precision="bf16-mixed",
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=csv_logger,
        accumulate_grad_batches=effective_batch_size // batch_size,
    )

    trainer.fit(model, dataloader)
