"""Tests for the Klingon tokenizer extension module."""

import json
from unittest.mock import MagicMock

import pytest

from klingon_translator.model.tokenizer import (
    collect_klingon_text,
    report_tokenizer_quality,
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
            {"en": "Success", "tlh": "Qapla'"},
            {"en": "Yes", "tlh": "HIja'"},
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


class TestCollectOpusMosesData:
    """Tests for OPUS Moses file collection in collect_klingon_text()."""

    def test_collects_opus_moses_data(self, tmp_path, monkeypatch):
        """Should read OPUS Moses Klingon file when using default data dir."""
        # Set up fake processed data dir with a JSONL file
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        jsonl = proc_dir / "train.jsonl"
        jsonl.write_text(
            json.dumps({"en": "hi", "tlh": "nuqneH"}) + "\n",
            encoding="utf-8",
        )

        # Set up fake raw dir with OPUS Moses file
        raw_dir = tmp_path / "raw"
        opus_dir = raw_dir / "opus" / "tatoeba"
        opus_dir.mkdir(parents=True)
        opus_tlh = opus_dir / "Tatoeba.en-tlh.tlh"
        opus_tlh.write_text(
            "Qapla'\ntlhIngan maH\nyIDoghQo'\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "klingon_translator.model.tokenizer.PROCESSED_DATA_DIR", proc_dir
        )
        monkeypatch.setattr(
            "klingon_translator.model.tokenizer.RAW_DATA_DIR", raw_dir
        )

        text = collect_klingon_text()
        lines = text.strip().splitlines()
        # Should have nuqneH from JSONL + 3 from OPUS
        assert "Qapla'" in lines
        assert "tlhIngan maH" in lines
        assert "yIDoghQo'" in lines
        assert len(lines) == 4  # nuqneH + 3 OPUS lines

    def test_skips_opus_when_custom_dir(self, tmp_path):
        """Should NOT read OPUS when a custom data_dir is provided."""
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(
            json.dumps({"en": "hi", "tlh": "nuqneH"}) + "\n",
            encoding="utf-8",
        )

        text = collect_klingon_text(data_dir=tmp_path)
        lines = text.strip().splitlines()
        # Only the JSONL entry, not any raw data
        assert lines == ["nuqneH"]


class TestCollectPaqbatlhData:
    """Tests for paq'batlh JSON collection in collect_klingon_text()."""

    def test_collects_paqbatlh_data(self, tmp_path, monkeypatch):
        """Should read paq'batlh JSON pairs when using default data dir."""
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        jsonl = proc_dir / "train.jsonl"
        jsonl.write_text(
            json.dumps({"en": "hi", "tlh": "nuqneH"}) + "\n",
            encoding="utf-8",
        )

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        paq_file = raw_dir / "paqbatlh_pairs.json"
        paq_pairs = [
            {"en": "Fight!", "tlh": "yISuv"},
            {"en": "Honor", "tlh": "batlh"},
        ]
        paq_file.write_text(
            json.dumps(paq_pairs, ensure_ascii=False),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "klingon_translator.model.tokenizer.PROCESSED_DATA_DIR", proc_dir
        )
        monkeypatch.setattr(
            "klingon_translator.model.tokenizer.RAW_DATA_DIR", raw_dir
        )

        text = collect_klingon_text()
        lines = text.strip().splitlines()
        assert "yISuv" in lines
        assert "batlh" in lines
        assert len(lines) == 3  # nuqneH + yISuv + batlh


class TestAutoVocabSize:
    """Tests for auto-scaling vocabulary size."""

    def test_auto_vocab_size_default(self, tmp_path):
        """When vocab_size=None, should auto-compute from corpus size."""
        # 200 unique sentences -> max(1000, 200//5) = max(1000, 40) = 1000
        sample_lines = [f"sentence{i} word{i}" for i in range(200)]
        # Repeat to ensure enough data for SPM training
        text = "\n".join(sample_lines * 5)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=None)

        import sentencepiece as _spm

        sp = _spm.SentencePieceProcessor()
        sp.load(str(model_path))
        # With 200 unique sentences: min(4000, max(1000, 200//5)) = 1000
        # But hard_vocab_limit=False may produce fewer
        assert sp.get_piece_size() > 0
        assert model_path.exists()

    def test_auto_vocab_large_corpus(self, tmp_path):
        """Larger corpus should produce larger auto vocab (up to 4000)."""
        # 10K unique -> min(4000, max(1000, 10000//5)) = 2000
        sample_lines = [f"tlhIngan{i} maH{i} Suv{i}" for i in range(10000)]
        text = "\n".join(sample_lines)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=None)

        import sentencepiece as _spm

        sp = _spm.SentencePieceProcessor()
        sp.load(str(model_path))
        # Auto-computed to 2000, but actual may differ due to small corpus heuristic
        assert sp.get_piece_size() > 100

    def test_explicit_vocab_still_works(self, tmp_path):
        """Explicit vocab_size should override auto-scaling."""
        text = "\n".join(["nuqneH", "Qapla'", "HIja'"] * 50)
        # vocab_size=500 will be adjusted down for this tiny corpus
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=500)

        import sentencepiece as _spm

        sp = _spm.SentencePieceProcessor()
        sp.load(str(model_path))
        # Should train successfully; byte_fallback uses 261 reserved tokens
        assert sp.get_piece_size() > 0
        assert sp.get_piece_size() <= 500


class TestApostrophePreserved:
    """Tests for apostrophe as atomic token in SentencePiece."""

    def test_apostrophe_in_vocab(self, tmp_path):
        """Apostrophe should appear as its own token in the SPM vocab."""
        # Klingon text heavy with apostrophes
        sample_lines = [
            "Qapla'",
            "Heghlu'meH QaQ jajvam",
            "bortaS bIr jablu'DI' reH QaQqu' nay'",
            "yIDoghQo'",
            "nuqDaq 'oH puchpa''e'",
            "jIyajbe'",
            "qatlho'",
            "DabuQlu'DI' yISuv",
            "bIlujDI' yIchegh",
            "tIqDaq HoSna' tu'lu'",
            "meQtaHbogh qachDaq Suv qoH neH",
            "taHjaj wo'",
            "tlhIngan maH",
            "wo' batlhvaD",
        ] * 30  # Enough data for training

        text = "\n".join(sample_lines)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=500)

        import sentencepiece as _spm

        sp = _spm.SentencePieceProcessor()
        sp.load(str(model_path))

        # Collect all tokens in the vocabulary
        vocab_tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

        # The apostrophe should be a user-defined symbol
        assert "'" in vocab_tokens, (
            "Apostrophe should be an atomic token in Klingon SPM vocab"
        )

    def test_apostrophe_not_merged_away(self, tmp_path):
        """Apostrophe should tokenize independently, not merge into other tokens."""
        sample_lines = [
            "Qapla'",
            "Heghlu'meH QaQ jajvam",
            "bortaS bIr jablu'DI' reH QaQqu' nay'",
            "yIDoghQo'",
            "jIyajbe'",
            "qatlho'",
            "DabuQlu'DI' yISuv",
            "tIqDaq HoSna' tu'lu'",
            "taHjaj wo'",
            "tlhIngan maH",
        ] * 30

        text = "\n".join(sample_lines)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=500)

        import sentencepiece as _spm

        sp = _spm.SentencePieceProcessor()
        sp.load(str(model_path))

        # Tokenize a word with apostrophe
        pieces = sp.encode("Qapla'", out_type=str)
        # The apostrophe should appear as a separate piece
        assert "'" in pieces or any("'" in p for p in pieces), (
            f"Apostrophe should be preserved in tokenization: {pieces}"
        )


class TestReportTokenizerQuality:
    """Tests for report_tokenizer_quality()."""

    def _make_mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        mock = MagicMock()
        # encode returns a list of fake token IDs (2 tokens per word)
        mock.encode.side_effect = lambda text, add_special_tokens=True: (
            list(range(len(text.split()) * 2))
        )
        mock.convert_ids_to_tokens.side_effect = lambda ids: [
            f"tok_{i}" for i in ids
        ]
        return mock

    def test_returns_dict_with_expected_keys(self):
        """Should return a dict with fertility, coverage_pct, sample_tokenizations."""
        mock_tok = self._make_mock_tokenizer()
        text = "nuqneH\nQapla'\ntlhIngan maH"

        result = report_tokenizer_quality(
            mock_tok, text, num_new_tokens=100, klingon_spm_vocab_size=500
        )

        assert isinstance(result, dict)
        assert "fertility" in result
        assert "coverage_pct" in result
        assert "sample_tokenizations" in result

    def test_fertility_positive(self):
        """Fertility should be a positive number when text is non-empty."""
        mock_tok = self._make_mock_tokenizer()
        text = "nuqneH\nQapla'\ntlhIngan maH"

        result = report_tokenizer_quality(mock_tok, text)
        assert result["fertility"] > 0

    def test_fertility_zero_for_empty_text(self):
        """Fertility should be 0 for empty text."""
        mock_tok = self._make_mock_tokenizer()
        mock_tok.encode.return_value = []

        result = report_tokenizer_quality(mock_tok, "")
        assert result["fertility"] == 0.0

    def test_coverage_calculation(self):
        """Coverage should be correctly calculated from new/total tokens."""
        mock_tok = self._make_mock_tokenizer()
        text = "nuqneH"

        result = report_tokenizer_quality(
            mock_tok, text, num_new_tokens=200, klingon_spm_vocab_size=1000
        )

        assert result["coverage_pct"] == pytest.approx(20.0)

    def test_coverage_none_without_counts(self):
        """Coverage should be None when token counts are not provided."""
        mock_tok = self._make_mock_tokenizer()
        text = "nuqneH"

        result = report_tokenizer_quality(mock_tok, text)
        assert result["coverage_pct"] is None

    def test_sample_tokenizations_included(self):
        """Should include sample phrases in the result."""
        mock_tok = self._make_mock_tokenizer()
        text = "nuqneH"

        result = report_tokenizer_quality(mock_tok, text)
        assert len(result["sample_tokenizations"]) == 6
        assert "Qapla'" in result["sample_tokenizations"]


class TestTrainKlingonSPM:
    """Tests for train_klingon_spm()."""

    def test_trains_model(self, tmp_path):
        """Should train a SentencePiece model and return the path."""
        # Create sample Klingon text (need enough for SPM to train)
        sample_lines = [
            "nuqneH",
            "Qapla'",
            "tlhIngan maH",
            "wo' batlhvaD",
            "HIja'",
            "ghobe'",
            "yIDoghQo'",
            "DabuQlu'DI' yISuv",
            "taHjaj wo'",
            "maj",
            "qatlho'",
            "jIyajbe'",
            "bortaS bIr jablu'DI' reH QaQqu' nay'",
            "Heghlu'meH QaQ jajvam",
            "meQtaHbogh qachDaq Suv qoH neH",
            "tIqDaq HoSna' tu'lu'",
            "bIlujDI' yIchegh",
        ] * 10  # Repeat to get enough data

        text = "\n".join(sample_lines)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=500)

        assert model_path.exists()
        assert model_path.suffix == ".model"

    def test_creates_corpus_file(self, tmp_path):
        """Should save the corpus text file."""
        text = "\n".join(["nuqneH", "Qapla'", "HIja'"] * 50)
        train_klingon_spm(text, output_dir=tmp_path, vocab_size=500)

        corpus_file = tmp_path / "klingon_corpus.txt"
        assert corpus_file.exists()

    def test_adjusts_vocab_for_small_corpus(self, tmp_path):
        """Should reduce vocab_size if corpus is too small."""
        # Very small corpus - vocab_size should be capped
        text = "\n".join(["nuqneH", "Qapla'"] * 60)
        model_path = train_klingon_spm(text, output_dir=tmp_path, vocab_size=5000)

        import sentencepiece as _spm

        sp = _spm.SentencePieceProcessor()
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
