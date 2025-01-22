# writing a pytorch lightnng based training script for smollm.
# Dataset: will be the cosmopedia-v2 dataset, with streaming enabled.
from model import SmolLM, SmolLMConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import tiktoken
import math
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger

ds = load_dataset(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True
)

dataloader = DataLoader(ds, batch_size=32, shuffle=True)

# lightning module


class SmolLMLightning(pl.LightningModule):
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.model = SmolLM(self.config)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        logits = self(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        # Log the loss with 4 decimal precision
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True
        )
        # Optionally print for observation
        # self.print(f"Train Loss: {loss.item():.4f}")
        return loss

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


# Training Script
if __name__ == "__main__":
    # Parameters
    block_size = 768
    batch_size = 16
    max_lr = 6e-4
    warmup_steps = 10
    max_steps = 25000

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    # Dataset and DataLoader
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = SmolLMLightning(SmolLMConfig())
    torch.set_float32_matmul_precision("high")

    # Set up TensorBoard logger
    logger = TensorBoardLogger("logs/", name="transformer_experiment")
    # tensorboard --logdir logs/
    # create your own theme!
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

    # Trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        callbacks=[LearningRateMonitor(logging_interval="step"), progress_bar],
        precision="bf16-mixed",  # 16-bit floating point, many other options are there
        log_every_n_steps=1,
        enable_progress_bar=True,  # show progress bar
        enable_model_summary=True,  # show model summary
        logger=logger,
    )

    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    # Training
    trainer.fit(model, dataloader)
