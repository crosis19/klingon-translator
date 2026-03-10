"""Tests for the Klingon tokenizer extension module."""

import json
import tempfile
from pathlib import Path

import pytest

from klingon_translator.model.tokenizer import (
    collect_klingon_text,
    train_klingon_spm,
)
from klingon_translator.utils.config import PROCESSED_DATA_DIR


class TestCollectKlingonText:
    """Tests for collect_klingon_text()."""

    def test_collects_from_processed_data(self):
        """Should collect Klingon text from JSONL files if they exist."""
        if not any(PROCESSED_DATA_DIR.glob("*.jsonl")):
            pytest.skip("No processed data available")

        text = collect_klingon_text()
        lines = text.strip().splitlines()
        assert len(lines) > 0, "Should collect at least some Klingon text"

    def test_returns_unique_lines(self):
        """Should deduplicate collected text."""
        if not any(PROCESSED_DATA_DIR.glob("*.jsonl")):
            pytest.skip("No processed data available")

        text = collect_klingon_text()
        lines = text.strip().splitlines()
        assert len(lines) == len(set(lines)), "Lines should be unique"

    def test_collects_from_custom_dir(self, tmp_path):
        """Should collect from a custom directory."""
        # Create a test JSONL file
        test_file = tmp_path / "test.jsonl"
        pairs = [
            {"en": "Hello", "tlh": "nuqneH"},
            {"en": "Success", "tlh": "Qapla\'"},
            {"en": "Yes", "tlh": "HIja\'"},
        ]
        with open(test_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        text = collect_klingon_text(data_dir=tmp_path)
        lines = text.strip().splitlines()
        assert len(lines) == 3
        assert "nuqneH" in lines

    def test_skips_empty_entries(self, tmp_path):
        """Should skip entries with empty Klingon text."""
        test_file = tmp_path / "test.jsonl"
        pairs = [
            {"en": "Hello", "tlh": "nuqneH"},
            {"en": "Empty", "tlh": ""},
            {"en": "Spaces", "tlh": "   "},
        ]
        with open(test_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        text = collect_klingon_text(data_dir=tmp_path)
        lines = text.strip().splitlines()
        assert len(lines) == 1


class TestTrainKlingonSPM:
    """Tests for train_klingon_spm()."""

    def test_trains_model(self, tmp_path):
        """Should train a SentencePiece model and return the path."""
        # Create sample Klingon text (need enough for SPM to train)
        sample_lines = [
            "nuqneH",
            "Qapla\'",
            "tlhIngan maH",
            "wo\' batlhvaD",
            "HIja\'",
            "ghobe\'",
            "yIDoghQo\'",
            "DabuQlu\'DI\' yISuv",
            "taHjaj wo\'",
            "maj",
            "qatlho\'",
            "jIyajbe\'",
            "bortaS bIr jablu\'DI\' reH QaQqu\' nay\'",
            "Heghlu\'meH QaQ jajvam",
            "meQtaHbogh qachDaq Suv qoH neH",
            "tIqDaq HoSna\' tu\'lu\'",
            "bIlujDI\' yIchegh",
        ] * 10  # Repeat to get enough data

        text = "\n".join(sample_lines)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=100)

        assert model_path.exists()
        assert model_path.suffix == ".model"

    def test_creates_corpus_file(self, tmp_path):
        """Should save the corpus text file."""
        text = "\n".join(["nuqneH", "Qapla\'", "HIja\'"] * 50)
        train_klingon_spm(text, output_dir=tmp_path, vocab_size=32)

        corpus_file = tmp_path / "klingon_corpus.txt"
        assert corpus_file.exists()

    def test_adjusts_vocab_for_small_corpus(self, tmp_path):
        """Should reduce vocab_size if corpus is too small."""
        # Very small corpus - vocab_size should be capped
        text = "\n".join(["nuqneH", "Qapla\'"] * 60)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=5000)

        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        # Should have been reduced well below 5000 for this tiny corpus
        assert sp.get_piece_size() < 500


class TestCollectKlingonTextIntegration:
    """Integration tests using real project data."""

    def test_real_data_coverage(self):
        """Verify we get enough text from real data for training."""
        if not any(PROCESSED_DATA_DIR.glob("*.jsonl")):
            pytest.skip("No processed data available")

        text = collect_klingon_text()
        num_lines = len(text.strip().splitlines())
        # We should have thousands of Klingon sentences
        assert num_lines > 1000, f"Expected >1000 lines, got {num_lines}"
        print(f"Collected {num_lines} unique Klingon sentences")
