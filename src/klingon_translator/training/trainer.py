"""Training loop setup and model saving utilities."""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from klingon_translator.training.gpu import GPUConfig


@dataclass
class TrainingConfig:
    """User-facing training hyperparameters.

    These are the knobs exposed at the top of the Colab notebook.

    Attributes:
        max_epochs: Maximum training epochs.
        early_stopping_patience: Stop if val loss does not improve for N evals.
        learning_rate: Peak learning rate.
        lr_scheduler: Schedule type. Options: linear, cosine,
            cosine_with_restarts, polynomial, constant, constant_with_warmup.
        warmup_steps: Number of warmup steps (0 to use warmup_ratio).
        warmup_ratio: Fraction of total steps for warmup.
        weight_decay: L2 regularization strength.
        seed: Random seed.
    """

    max_epochs: int = 15
    early_stopping_patience: int = 3
    learning_rate: float = 2e-5
    lr_scheduler: str = "linear"
    warmup_steps: int = 200
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    seed: int = 42


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    val_dataset: Dataset,
    gpu_config: GPUConfig,
    training_config: TrainingConfig,
    output_dir: str | Path = "/content/checkpoints",
) -> Seq2SeqTrainer:
    """Build a Seq2SeqTrainer with GPU-adaptive settings.

    Constructs Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
    and EarlyStoppingCallback, then assembles them into a trainer.

    Args:
        model: The extended/fine-tuned model.
        tokenizer: The extended tokenizer.
        train_dataset: Pre-tokenized training dataset.
        val_dataset: Pre-tokenized validation dataset.
        gpu_config: Hardware-adaptive settings from detect_gpu().
        training_config: User-specified hyperparameters.
        output_dir: Directory for checkpoints.

    Returns:
        Configured Seq2SeqTrainer ready for .train().
    """
    import gc

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.max_epochs,
        per_device_train_batch_size=gpu_config.batch_size,
        per_device_eval_batch_size=gpu_config.batch_size,
        gradient_accumulation_steps=gpu_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type=training_config.lr_scheduler,
        warmup_steps=training_config.warmup_steps,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        bf16=gpu_config.use_bf16,
        fp16=gpu_config.use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=False,
        logging_steps=50,
        report_to="none",
        seed=training_config.seed,
        dataloader_num_workers=gpu_config.dataloader_num_workers,
        dataloader_pin_memory=gpu_config.dataloader_pin_memory,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=training_config.early_stopping_patience
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[early_stopping],
    )

    # Print training summary
    tc = training_config
    gc_cfg = gpu_config
    print("Trainer ready:")
    patience = tc.early_stopping_patience
    print(f"  Epochs: up to {tc.max_epochs} (early stop={patience})")
    bs = gc_cfg.batch_size
    accum = gc_cfg.gradient_accumulation_steps
    eff = gc_cfg.effective_batch_size
    print(f"  Batch: {bs} x {accum} accum = {eff} effective")
    print(f"  LR: {tc.learning_rate} ({tc.lr_scheduler} schedule)")
    if tc.warmup_steps:
        warmup_msg = f"{tc.warmup_steps} steps"
    else:
        warmup_msg = f"{tc.warmup_ratio:.0%} of total"
    print(f"  Warmup: {warmup_msg}")
    print(f"  Weight decay: {tc.weight_decay}")
    precision = "BF16" if gc_cfg.use_bf16 else "FP16" if gc_cfg.use_fp16 else "FP32"
    print(f"  Precision: {precision}")

    return trainer


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    drive_dir: str | Path,
    local_dir: str | Path | None = "/content/fine-tuned",
) -> Path:
    """Save model to Drive and optionally to a local fast-access copy.

    Args:
        model: The trained model.
        tokenizer: The extended tokenizer.
        drive_dir: Google Drive path for persistent storage.
        local_dir: Local SSD path for fast re-loading. None to skip.

    Returns:
        Path to the Drive save directory.
    """
    drive_dir = Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)

    print("Saving fine-tuned model to Google Drive...")
    t0 = time.time()
    model.save_pretrained(str(drive_dir))
    tokenizer.save_pretrained(str(drive_dir))
    print(f"  Saved to {drive_dir} in {time.time() - t0:.0f}s")

    if local_dir is not None:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_dir))
        tokenizer.save_pretrained(str(local_dir))
        print(f"  Also saved local copy to {local_dir}")

    return drive_dir
