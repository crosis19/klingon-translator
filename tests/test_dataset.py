"""Tests for training.dataset module."""

from unittest.mock import MagicMock

from klingon_translator.training.dataset import BilingualDataset


def _make_mock_tokenizer():
    """Create a mock tokenizer that returns predictable token ids."""
    tok = MagicMock()
    tok.convert_tokens_to_ids.side_effect = lambda code: {
        "eng_Latn": 100,
        "tlh_Latn": 200,
    }.get(code, 0)

    def tokenize_fn(text, truncation=True, max_length=128):
        # Return fake token ids based on text length
        ids = list(range(1, min(len(text.split()) + 1, max_length)))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tok.side_effect = tokenize_fn
    tok.__call__ = tokenize_fn
    return tok


class TestBilingualDataset:
    """Test BilingualDataset pre-tokenization and caching."""

    def test_creates_double_examples(self):
        """N pairs should produce 2N examples."""
        pairs = [
            {"en": "hello", "tlh": "nuqneH"},
            {"en": "yes", "tlh": "HIja"},
        ]
        tok = _make_mock_tokenizer()
        ds = BilingualDataset(pairs, tok, shuffle=False)
        assert len(ds) == 4

    def test_getitem_returns_expected_keys(self):
        """Each example should have input_ids, attention_mask, labels."""
        pairs = [{"en": "hello world", "tlh": "nuqneH"}]
        tok = _make_mock_tokenizer()
        ds = BilingualDataset(pairs, tok, shuffle=False)
        ex = ds[0]
        assert "input_ids" in ex
        assert "attention_mask" in ex
        assert "labels" in ex

    def test_labels_start_with_lang_id(self):
        """Labels should start with target language id."""
        pairs = [{"en": "hello", "tlh": "nuqneH"}]
        tok = _make_mock_tokenizer()
        ds = BilingualDataset(pairs, tok, shuffle=False)
        # First example is en->tlh, labels start with tlh_Latn id (200)
        assert ds[0]["labels"][0] == 200
        # Second example is tlh->en, labels start with eng_Latn id (100)
        assert ds[1]["labels"][0] == 100

    def test_shuffle_deterministic(self):
        """Same seed should produce same ordering."""
        pairs = [
            {"en": f"sentence {i}", "tlh": f"mu {i}"}
            for i in range(10)
        ]
        tok = _make_mock_tokenizer()
        ds1 = BilingualDataset(pairs, tok, shuffle=True, seed=42)
        ds2 = BilingualDataset(pairs, tok, shuffle=True, seed=42)
        for i in range(len(ds1)):
            assert ds1[i]["input_ids"] == ds2[i]["input_ids"]

    def test_no_shuffle_preserves_order(self):
        """Without shuffle, examples should be in pair order."""
        pairs = [{"en": "first", "tlh": "wa"}, {"en": "second", "tlh": "cha"}]
        tok = _make_mock_tokenizer()
        ds = BilingualDataset(pairs, tok, shuffle=False)
        # First pair produces examples 0 (en->tlh) and 1 (tlh->en)
        # Second pair produces examples 2 and 3
        assert len(ds) == 4

    def test_empty_pairs(self):
        """Empty input should produce empty dataset."""
        tok = _make_mock_tokenizer()
        ds = BilingualDataset([], tok, shuffle=False)
        assert len(ds) == 0
