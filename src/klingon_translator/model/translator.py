"""Core KlingonTranslator class - the main interface for translation."""

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from klingon_translator.utils.config import (
    BASE_MODEL_ID,
    ENGLISH_CODE,
    KLINGON_CODE,
    MODELS_DIR,
)


class KlingonTranslator:
    """English <-> Klingon translator using a fine-tuned NLLB-200 model.

    Usage:
        translator = KlingonTranslator()  # loads base model
        translator = KlingonTranslator("path/to/fine-tuned")  # loads fine-tuned model

        result = translator.translate("Hello, world!", src_lang="eng_Latn", tgt_lang="tlh_Latn")
        result = translator.to_klingon("Hello, world!")
        result = translator.to_english("nuqneH")
    """

    def __init__(self, model_path: str | Path | None = None):
        """Load model and tokenizer.

        Args:
            model_path: Path to a fine-tuned model directory, or None to load the
                base NLLB-200 model from Hugging Face (useful for testing the pipeline
                before fine-tuning).
        """
        model_id = str(model_path) if model_path else BASE_MODEL_ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def translate(
        self,
        text: str,
        src_lang: str = ENGLISH_CODE,
        tgt_lang: str = KLINGON_CODE,
        max_length: int = 128,
    ) -> str:
        """Translate a single string.

        Args:
            text: Input text to translate.
            src_lang: Source language code (NLLB format).
            tgt_lang: Target language code (NLLB format).
            max_length: Maximum output token length.

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
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str = ENGLISH_CODE,
        tgt_lang: str = KLINGON_CODE,
        max_length: int = 128,
    ) -> list[str]:
        """Translate a batch of strings.

        Args:
            texts: List of input texts.
            src_lang: Source language code.
            tgt_lang: Target language code.
            max_length: Maximum output token length.

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
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def to_klingon(self, text: str) -> str:
        """Shortcut: translate English to Klingon."""
        return self.translate(text, src_lang=ENGLISH_CODE, tgt_lang=KLINGON_CODE)

    def to_english(self, text: str) -> str:
        """Shortcut: translate Klingon to English."""
        return self.translate(text, src_lang=KLINGON_CODE, tgt_lang=ENGLISH_CODE)

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
        return save_dir


if __name__ == "__main__":
    # Quick test with base model (won't produce real Klingon, but validates the pipeline)
    print("Loading base NLLB-200 model (this may take a minute)...")
    t = KlingonTranslator()
    print(f"Model loaded on {t.device}")
    result = t.translate("Hello, how are you?", tgt_lang="fra_Latn")
    print(f"English -> French (sanity check): {result}")
    print("Pipeline works! Fine-tuning needed for Klingon support.")
