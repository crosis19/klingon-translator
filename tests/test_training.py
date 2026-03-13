"""Tests for training.trainer module."""

from klingon_translator.training.gpu import GPUConfig
from klingon_translator.training.trainer import TrainingConfig


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.max_epochs == 15
        assert config.early_stopping_patience == 3
        assert config.learning_rate == 2e-5
        assert config.lr_scheduler == "linear"
        assert config.warmup_steps == 200
        assert config.warmup_ratio == 0.0
        assert config.weight_decay == 0.01
        assert config.seed == 42

    def test_custom_values(self):
        config = TrainingConfig(
            max_epochs=30,
            early_stopping_patience=5,
            learning_rate=1e-4,
            lr_scheduler="cosine",
            warmup_steps=0,
            warmup_ratio=0.1,
            weight_decay=0.05,
            seed=123,
        )
        assert config.max_epochs == 30
        assert config.lr_scheduler == "cosine"
        assert config.warmup_ratio == 0.1


class TestGPUConfigCompatibility:
    """Test that GPUConfig works with TrainingConfig."""

    def test_configs_are_independent(self):
        """GPU and training configs should be separate concerns."""
        gpu = GPUConfig(
            gpu_name="Test",
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
        tc = TrainingConfig(learning_rate=1e-4)
        # Both exist independently
        assert gpu.batch_size == 4
        assert tc.learning_rate == 1e-4
