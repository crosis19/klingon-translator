"""GPU detection and adaptive training configuration."""

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GPUConfig:
    """Training hardware configuration.

    All values are set explicitly by the user in the notebook.
    Use ``get_gpu_info()`` to see what hardware is available,
    then construct a GPUConfig with the settings you want.

    Attributes:
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation factor.
        use_bf16: Whether to use BF16 mixed precision.
        use_fp16: Whether to use FP16 mixed precision.
        gradient_checkpointing: Trade compute for memory.
        dataloader_num_workers: Number of dataloader workers.
        dataloader_pin_memory: Whether to pin memory in dataloader.
    """

    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    use_bf16: bool = True
    use_fp16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    @property
    def effective_batch_size(self) -> int:
        """batch_size * gradient_accumulation_steps."""
        return self.batch_size * self.gradient_accumulation_steps


def get_gpu_info() -> dict[str, object]:
    """Report available GPU hardware (does NOT set any config).

    Call this to see what GPU you have, then set GPUConfig
    values accordingly.

    Returns:
        Dict with gpu_name, gpu_memory_gb, and cuda_available.
    """
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "gpu_name": "CPU",
            "gpu_memory_gb": 0.0,
        }
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory
            / 1e9,
            1,
        ),
    }


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
