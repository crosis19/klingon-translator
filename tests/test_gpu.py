"""Tests for training.gpu module."""

from unittest.mock import MagicMock, patch

from klingon_translator.training.gpu import (
    GPUConfig,
    detect_gpu,
    enable_tf32_if_available,
    set_seed,
)


class TestGPUConfig:
    """Test GPUConfig dataclass."""

    def test_dataclass_fields(self):
        config = GPUConfig(
            gpu_name="Test GPU",
            gpu_memory_gb=16.0,
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
        assert config.gpu_name == "Test GPU"
        assert config.gpu_memory_gb == 16.0
        assert config.effective_batch_size == 32


class TestDetectGPU:
    """Test GPU detection and adaptive config."""

    @patch("klingon_translator.training.gpu.torch")
    def test_cpu_fallback(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        config = detect_gpu()
        assert config.gpu_name == "CPU"
        assert config.gpu_memory_gb == 0.0
        assert config.is_a100 is False
        assert config.gradient_checkpointing is True
        assert config.use_bf16 is False
        assert config.use_fp16 is False

    @patch("klingon_translator.training.gpu.torch")
    def test_a100_config(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-80GB"
        mock_props = MagicMock()
        mock_props.total_memory = 80e9  # 80 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        config = detect_gpu()
        assert config.is_a100 is True
        assert config.batch_size == 32
        assert config.use_bf16 is True
        assert config.use_fp16 is False
        assert config.gradient_checkpointing is False
        assert config.dataloader_num_workers == 4

    @patch("klingon_translator.training.gpu.torch")
    def test_t4_config(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla T4"
        mock_props = MagicMock()
        mock_props.total_memory = 16e9  # 16 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        config = detect_gpu()
        assert config.is_a100 is False
        assert config.batch_size == 4
        assert config.use_bf16 is False
        assert config.use_fp16 is True
        assert config.gradient_checkpointing is True
        assert config.gradient_accumulation_steps == 8


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
