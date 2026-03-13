"""Core KlingonTranslator class - the main interface for translation."""

import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from klingon_translator.utils.config import (
    BASE_MODEL_ID,
    ENGLISH_CODE,
    KLINGON_CODE,
    MODELS_DIR,
)


def clean_translation(text: str) -> str:
    """Post-process translated text to fix tokenizer spacing artifacts.

    Subword tokenizers (especially extended ones) can introduce
    spurious spaces when tokens from different vocabularies are
    combined during decoding.  This function cleans up common
    issues without altering the actual content.

    Args:
        text: Raw decoded translation string.

    Returns:
        Cleaned translation string.
    """
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    # Remove spaces before apostrophes  (e.g. "Qapla '" -> "Qapla'")
    text = re.sub(r" '", "'", text)
    # Remove spaces after apostrophes when followed by a letter
    # (e.g. "jablu' DI'" -> "jablu'DI'")
    text = re.sub(r"' (?=[A-Za-z])", "'", text)
    # Remove spaces before punctuation  (e.g. "nuqneH ?" -> "nuqneH?")
    text = re.sub(r" ([?.!,;:])", r"\1", text)
    # Remove spaces after opening quotes/parens
    text = re.sub(r'(["\(]) ', r"\1", text)
    # Remove spaces before closing quotes/parens
    text = re.sub(r' (["\)])', r"\1", text)
    return text.strip()

# Default path for the extended (Klingon-ready) model
EXTENDED_MODEL_DIR = MODELS_DIR / "nllb-klingon-extended"


class KlingonTranslator:
    """English <-> Klingon translator using a fine-tuned NLLB-200 model.

    Usage:
        # Load the extended (pre-fine-tuning) model
        translator = KlingonTranslator()

        # Load a specific fine-tuned checkpoint
        translator = KlingonTranslator("path/to/fine-tuned")

        # Translate
        result = translator.to_klingon("Hello, world!")
        result = translator.to_english("nuqneH")
    """

    def __init__(self, model_path: str | Path | None = None):
        """Load model and tokenizer.

        Resolution order for model_path=None:
        1. Extended model (models/nllb-klingon-extended) if it exists
        2. Base NLLB-200 from Hugging Face (no Klingon support)

        Args:
            model_path: Path to a model directory, or None for auto-detection.
        """
        if model_path is not None:
            model_id = str(model_path)
            source = f"custom: {model_id}"
        elif EXTENDED_MODEL_DIR.exists():
            model_id = str(EXTENDED_MODEL_DIR)
            source = f"extended: {EXTENDED_MODEL_DIR}"
        else:
            model_id = BASE_MODEL_ID
            source = f"base: {BASE_MODEL_ID} (no Klingon support)"

        print(f"Loading model ({source})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"  Ready on {self.device} (vocab={len(self.tokenizer)})")

    @property
    def has_klingon(self) -> bool:
        """Check if the loaded model supports Klingon translation."""
        return KLINGON_CODE in self.tokenizer.get_vocab()

    def translate(
        self,
        text: str,
        src_lang: str = ENGLISH_CODE,
        tgt_lang: str = KLINGON_CODE,
        max_length: int = 128,
        num_beams: int = 5,
    ) -> str:
        """Translate a single string.

        Args:
            text: Input text to translate.
            src_lang: Source language code (NLLB format).
            tgt_lang: Target language code (NLLB format).
            max_length: Maximum output token length.
            num_beams: Beam search width (higher = better but slower).

        Returns:
            Translated text string.
        """
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(
            self.device
        )
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=num_beams,
            )

        raw = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        return clean_translation(raw)

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str = ENGLISH_CODE,
        tgt_lang: str = KLINGON_CODE,
        max_length: int = 128,
        num_beams: int = 5,
    ) -> list[str]:
        """Translate a batch of strings.

        Args:
            texts: List of input texts.
            src_lang: Source language code.
            tgt_lang: Target language code.
            max_length: Maximum output token length.
            num_beams: Beam search width.

        Returns:
            List of translated strings.
        """
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=num_beams,
            )

        raw = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return [clean_translation(t) for t in raw]

    def to_klingon(self, text: str, **kwargs) -> str:
        """Shortcut: translate English to Klingon."""
        return self.translate(
            text, src_lang=ENGLISH_CODE, tgt_lang=KLINGON_CODE, **kwargs
        )

    def to_english(self, text: str, **kwargs) -> str:
        """Shortcut: translate Klingon to English."""
        return self.translate(
            text, src_lang=KLINGON_CODE, tgt_lang=ENGLISH_CODE, **kwargs
        )

    def save(self, path: str | Path | None = None) -> Path:
        """Save model and tokenizer to disk.

        Args:
            path: Directory to save to. Defaults to models/fine-tuned.

        Returns:
            Path where the model was saved.
        """
        save_dir = Path(path) if path else MODELS_DIR / "fine-tuned"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Saved model to: {save_dir}")
        return save_dir


if __name__ == "__main__":
    translator = KlingonTranslator()

    if translator.has_klingon:
        print("\nKlingon support detected! Testing translation...")
        result = translator.to_klingon("Hello, how are you?")
        print(f"  English -> Klingon: {result}")
        result = translator.to_english("nuqneH")
        print(f"  Klingon -> English: {result}")
    else:
        print("\nNo Klingon support (base model). Testing with French...")
        result = translator.translate("Hello, how are you?", tgt_lang="fra_Latn")
        print(f"  English -> French: {result}")
        print("\nRun tokenizer extension first:")
        print("  python -m klingon_translator.model.tokenizer")
