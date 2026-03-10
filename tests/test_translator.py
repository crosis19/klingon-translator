"""Basic smoke tests for the klingon_translator package."""

import pytest

from klingon_translator.utils.config import (
    BASE_MODEL_ID,
    DATA_DIR,
    ENGLISH_CODE,
    KLINGON_CODE,
    MODELS_DIR,
    PROJECT_ROOT,
    ensure_dirs,
)


def test_package_imports():
    """Verify the package can be imported."""
    import klingon_translator

    assert klingon_translator.__version__ == "0.1.0"


def test_config_paths_exist():
    """Verify config paths are reasonable."""
    assert PROJECT_ROOT.exists()
    assert "Klingon Translator" in str(PROJECT_ROOT)


def test_config_constants():
    """Verify config constants are set."""
    assert BASE_MODEL_ID == "facebook/nllb-200-distilled-600M"
    assert ENGLISH_CODE == "eng_Latn"
    assert KLINGON_CODE == "tlh_Latn"


def test_ensure_dirs():
    """Verify ensure_dirs creates directories."""
    ensure_dirs()
    assert DATA_DIR.exists()
    assert MODELS_DIR.exists()


def test_translator_class_importable():
    """Verify KlingonTranslator can be imported (doesn\'t load model)."""
    from klingon_translator.model.translator import KlingonTranslator

    assert KlingonTranslator is not None


def test_translator_has_extended_model_dir():
    """Verify EXTENDED_MODEL_DIR is defined."""
    from klingon_translator.model.translator import EXTENDED_MODEL_DIR

    assert EXTENDED_MODEL_DIR is not None
    assert "nllb-klingon-extended" in str(EXTENDED_MODEL_DIR)


def test_data_module_importable():
    """Verify data module can be imported."""
    from klingon_translator.data.download import (
        build_dataset,
        download_tatoeba,
        load_proverbs,
        parse_boqwi,
    )

    assert all(callable(f) for f in [download_tatoeba, parse_boqwi, load_proverbs, build_dataset])


def test_tokenizer_module_importable():
    """Verify tokenizer module can be imported."""
    from klingon_translator.model.tokenizer import (
        collect_klingon_text,
        extend_nllb_tokenizer,
        run_pipeline,
        train_klingon_spm,
    )

    assert all(callable(f) for f in [collect_klingon_text, train_klingon_spm, extend_nllb_tokenizer, run_pipeline])
