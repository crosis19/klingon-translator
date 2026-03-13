"""GPU detection and adaptive training configuration."""

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GPUConfig:
    """Hardware-adaptive training configuration.

    Attributes:
        gpu_name: Name of the detected GPU (or "CPU").
        gpu_memory_gb: Total GPU memory in GB.
        is_a100: Whether the GPU has >=40GB VRAM (A100-class).
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation factor.
        effective_batch_size: batch_size * gradient_accumulation_steps.
        use_bf16: Whether to use BF16 mixed precision.
        use_fp16: Whether to use FP16 mixed precision.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        dataloader_num_workers: Number of dataloader workers.
        dataloader_pin_memory: Whether to pin memory in dataloader.
    """

    gpu_name: str
    gpu_memory_gb: float
    is_a100: bool
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    use_bf16: bool
    use_fp16: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool


def detect_gpu(**overrides: object) -> GPUConfig:
    """Detect GPU hardware and return an adaptive training configuration.

    Returns A100-optimized config (batch 32, bf16, no grad checkpointing)
    or T4-conservative config (batch 4, fp16, gradient checkpointing).
    Falls back to CPU config if no GPU is available.

    Any keyword argument matching a GPUConfig field will override the
    auto-detected value.  If ``batch_size`` or
    ``gradient_accumulation_steps`` is overridden, ``effective_batch_size``
    is automatically recalculated unless it is also explicitly provided.

    Args:
        **overrides: Optional field overrides, e.g.
            ``detect_gpu(batch_size=8, gradient_accumulation_steps=4)``.

    Returns:
        GPUConfig with all hardware-adaptive settings.
    """
    if not torch.cuda.is_available():
        defaults = dict(
            gpu_name="CPU",
            gpu_memory_gb=0.0,
            is_a100=False,
            batch_size=2,
            gradient_accumulation_steps=16,
            effective_batch_size=32,
            use_bf16=False,
            use_fp16=False,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        is_a100 = gpu_mem >= 40

        if is_a100:
            defaults = dict(
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_mem,
                is_a100=True,
                batch_size=32,
                gradient_accumulation_steps=1,
                effective_batch_size=32,
                use_bf16=True,
                use_fp16=False,
                gradient_checkpointing=False,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
            )
        else:
            defaults = dict(
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_mem,
                is_a100=False,
                batch_size=4,
                gradient_accumulation_steps=8,
                effective_batch_size=32,
                use_bf16=False,
                use_fp16=True,
                gradient_checkpointing=True,
                dataloader_num_workers=2,
                dataloader_pin_memory=False,
            )

    # Apply user overrides
    merged = {**defaults, **overrides}

    # Recalculate effective batch size when batch params change
    if "effective_batch_size" not in overrides and (
        "batch_size" in overrides
        or "gradient_accumulation_steps" in overrides
    ):
        merged["effective_batch_size"] = (
            merged["batch_size"]
            * merged["gradient_accumulation_steps"]
        )

    return GPUConfig(**merged)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_tf32_if_available() -> bool:
    """Enable TF32 matmul on A100+ GPUs for faster float32 operations.

    TF32 uses TensorCores to accelerate float32 matrix multiplications
    by ~3x with minimal precision loss. Only available on Ampere+ GPUs.

    Returns:
        True if TF32 was enabled, False otherwise.
    """
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem >= 40:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return True
    return False
