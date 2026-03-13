"""Tests for training.evaluate module."""

from klingon_translator.training.evaluate import (
    DEFAULT_SAMPLE_PHRASES_EN,
    DEFAULT_SAMPLE_PHRASES_TLH,
    EvalScores,
    SampleResult,
    translate_batch,
)


class TestEvalScores:
    """Test EvalScores dataclass."""

    def test_dataclass_fields(self):
        scores = EvalScores(
            bleu_en2tlh=10.5,
            bleu_tlh2en=12.3,
            chrf_en2tlh=30.1,
            chrf_tlh2en=35.2,
            bleu_average=11.4,
            chrf_average=32.65,
        )
        assert scores.bleu_en2tlh == 10.5
        assert scores.bleu_average == 11.4
        assert scores.chrf_average == 32.65


class TestSampleResult:
    """Test SampleResult dataclass."""

    def test_match_true_when_equal(self):
        result = SampleResult(
            input="hello",
            expected="nuqneH",
            predicted="nuqneH",
            match=True,
        )
        assert result.match is True

    def test_match_false_when_different(self):
        result = SampleResult(
            input="hello",
            expected="nuqneH",
            predicted="Qapla",
            match=False,
        )
        assert result.match is False


class TestDefaultSamplePhrases:
    """Test default sample phrase lists."""

    def test_en_phrases_nonempty(self):
        assert len(DEFAULT_SAMPLE_PHRASES_EN) > 0

    def test_tlh_phrases_nonempty(self):
        assert len(DEFAULT_SAMPLE_PHRASES_TLH) > 0

    def test_phrases_are_tuples(self):
        for phrase, expected in DEFAULT_SAMPLE_PHRASES_EN:
            assert isinstance(phrase, str)
            assert isinstance(expected, str)
        for phrase, expected in DEFAULT_SAMPLE_PHRASES_TLH:
            assert isinstance(phrase, str)
            assert isinstance(expected, str)


class TestTranslateBatch:
    """Test translate_batch function."""

    def test_empty_input_returns_empty(self):
        result = translate_batch(
            [], None, None, "eng_Latn", "tlh_Latn"
        )
        assert result == []
