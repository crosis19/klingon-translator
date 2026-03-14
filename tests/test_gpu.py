"""Tests for training.gpu module."""

from unittest.mock import MagicMock, patch

from klingon_translator.training.gpu import (
    GPUConfig,
    enable_tf32_if_available,
    get_gpu_info,
    set_seed,
)


class TestGPUConfig:
    """Test GPUConfig dataclass."""

    def test_defaults(self):
        config = GPUConfig()
        assert config.batch_size == 32
        assert config.gradient_accumulation_steps == 1
        assert config.use_bf16 is True
        assert config.use_fp16 is False
        assert config.gradient_checkpointing is False
        assert config.dataloader_num_workers == 4
        assert config.dataloader_pin_memory is True

    def test_effective_batch_size(self):
        config = GPUConfig(batch_size=4, gradient_accumulation_steps=8)
        assert config.effective_batch_size == 32

    def test_custom_values(self):
        config = GPUConfig(
            batch_size=8,
            gradient_accumulation_steps=4,
            use_bf16=False,
            use_fp16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=False,
        )
        assert config.batch_size == 8
        assert config.use_fp16 is True
        assert config.gradient_checkpointing is True
        assert config.effective_batch_size == 32


class TestGetGPUInfo:
    """Test GPU info reporting."""

    @patch("klingon_translator.training.gpu.torch")
    def test_cpu_fallback(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        info = get_gpu_info()
        assert info["cuda_available"] is False
        assert info["gpu_name"] == "CPU"
        assert info["gpu_memory_gb"] == 0.0

    @patch("klingon_translator.training.gpu.torch")
    def test_gpu_detected(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_props = MagicMock()
        mock_props.total_memory = 80e9
        mock_torch.cuda.get_device_properties.return_value = mock_props
        info = get_gpu_info()
        assert info["cuda_available"] is True
        assert info["gpu_name"] == "NVIDIA A100"
        assert info["gpu_memory_gb"] == 80.0


class TestSetSeed:
    """Test reproducibility seeding."""

    def test_reproducibility(self):
        import random

        set_seed(42)
        val1 = random.random()
        set_seed(42)
        val2 = random.random()
        assert val1 == val2

    def test_different_seeds_differ(self):
        import random

        set_seed(42)
        val1 = random.random()
        set_seed(99)
        val2 = random.random()
        assert val1 != val2


class TestEnableTF32:
    """Test TF32 enablement."""

    @patch("klingon_translator.training.gpu.torch")
    def test_no_gpu_returns_false(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        assert enable_tf32_if_available() is False

    @patch("klingon_translator.training.gpu.torch")
    def test_small_gpu_returns_false(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 16e9
        mock_torch.cuda.get_device_properties.return_value = mock_props
        assert enable_tf32_if_available() is False

    @patch("klingon_translator.training.gpu.torch")
    def test_a100_returns_true(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 80e9
        mock_torch.cuda.get_device_properties.return_value = mock_props
        assert enable_tf32_if_available() is True
