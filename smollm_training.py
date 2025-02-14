# import for  colab/kaggle
# !pip install datasets transformers wandb -q
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
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint

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


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_heads
        self.n_embd = config.n_embed

        # Linear projections for Q, K, V
        # self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # [n_embd, 3 * n_embd]
        self.w_q = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.w_k = nn.Linear(
            config.n_embed, config.n_embed // config.n_key_value_heads, bias=False
        )
        self.w_v = nn.Linear(
            config.n_embed, config.n_embed // config.n_key_value_heads, bias=False
        )
        self.c_proj = nn.Linear(
            config.n_embed, config.n_embed, bias=False
        )  # [n_embd, n_embd]
        self.c_proj.NANGPT_SCALE_INIT = 1

        self.n_rep = self.config.n_heads // self.config.n_key_value_heads

        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # [B, T, n_embd]

        # Linear projection and split into Q, K, V
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # [B, T, n_embd] each
        q = self.w_q(x)  # [B, T, 576]
        k = self.w_k(x)  # [B, T, 192]
        v = self.w_v(x)  # [B, T, 192]

        # Reshape for multi-head attention
        k = k.view(
            B,
            T,
            self.config.n_key_value_heads,
            k.size(-1) // self.config.n_key_value_heads,
        ).transpose(
            1, 2
        )  # [B, 3, T, 64]
        q = q.view(
            B, T, self.config.n_heads, q.size(-1) // self.config.n_heads
        ).transpose(
            1, 2
        )  # [B, 9, T, 64]
        v = v.view(
            B,
            T,
            self.config.n_key_value_heads,
            v.size(-1) // self.config.n_key_value_heads,
        ).transpose(
            1, 2
        )  # [B, 3, T, 64]

        # repeat k and v for each head
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # # Attention scores
        # att = (q @ k.transpose(-2, -1)) * (
        #     1.0 / (k.size(-1) ** 0.5)
        # )  # [B, n_head, T, T]
        # att = att.masked_fill(
        #     self.bias[:, :, :T, :T] == 0, float("-inf")
        # )  # [B, n_head, T, T]
        # att = F.softmax(att, dim=-1)  # [B, n_head, T, T]

        # # Weighted sum of values
        # y = att @ v  # [B, n_head, T, n_embd/n_head]

        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash attention
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, n_embd]
        y = self.c_proj(y)  # [B, T, n_embd]
        y = self.resid_dropout(y)  # [B, T, n_embd]

        return y


class MLP(nn.Module):

    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.mlp_hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(config.mlp_hidden_dim, config.n_embed, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        return x


class LlamaMLP(nn.Module):

    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.hidden_dim = config.mlp_hidden_dim  # 1536
        self.w1 = nn.Linear(config.n_embed, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, self.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderBlockWithRMSNorm(nn.Module):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
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
        self.wte = nn.Embedding(
            config.vocab_size, config.n_embed
        )  # [vocab_size, n_embd]
        self.wpe = nn.Embedding(
            config.block_size, config.n_embed
        )  # [max_seq_len, n_embd]
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlockWithRMSNorm(config) for _ in range(config.n_layers)]
        )
        self.rms_norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)  # [n_embd]
        self.lm_head = nn.Linear(
            config.n_embed, config.vocab_size, bias=False
        )  # [n_embd, vocab_size]

        # weight sharing
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.wpe(pos)  # position embeddings of shape (T, n_embd)
        x = self.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = x + pos_emb

        # forward the blocks of the transformer
        for block in self.blocks:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.rms_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text given a starting sequence of tokens.

        Args:
            idx (torch.Tensor): Starting token indices, shape (B, T)
            max_new_tokens (int): Number of tokens to generate
            temperature (float): Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
            top_k (int): If specified, only sample from the top k most probable tokens
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class SmolLMLightning(pl.LightningModule):
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

        # Log the loss with 4 decimal precision
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True
        )

        # Generate text every n steps, but only if we're not already generating
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
            generated_ids = self.model.generate(
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

            # Log to WandB
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {"generated_text": generated_text, "global_step": self.global_step}
                )
        except Exception as e:
            print(f"Generation failed with error: {str(e)}")

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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    dataloader = load_cosmopedia_dataset(batch_size=batch_size, seq_length=block_size)

    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best-checkpoint.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = SmolLMLightning.load_from_checkpoint(
            checkpoint_path,
            config=SmolLMConfig(),
            lr=max_lr,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
        )
    else:
        print("Starting training from scratch")
        model = SmolLMLightning(SmolLMConfig(), max_lr, warmup_steps, max_steps)

    # Replace TensorBoard logger with WandB logger
    wandb_logger = WandbLogger(
        project="smollm",  # your project name
        name="transformer_experiment",  # name of the run
        log_model=True,  # log model checkpoints
    )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-checkpoint",
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
        logger=wandb_logger,
        accumulate_grad_batches=effective_batch_size // batch_size,
    )

    trainer.fit(model, dataloader)
