"""Klingon tokenizer extension for NLLB-200.

Since NLLB doesn't include Klingon, we need to:
1. Train a SentencePiece model on available Klingon text
2. Merge new Klingon subword tokens into NLLB's tokenizer
3. Register the tlh_Latn language code
4. Initialize new token embeddings from English embeddings (transfer learning)
"""

import json
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from klingon_translator.utils.config import (
    BASE_MODEL_ID,
    ENGLISH_CODE,
    KLINGON_CODE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)


def collect_klingon_text(data_dir: Path | None = None) -> str:
    """Collect all available Klingon text for tokenizer training.

    Args:
        data_dir: Directory containing processed .jsonl files.

    Returns:
        Concatenated Klingon text, one sentence per line.
    """
    data_dir = data_dir or PROCESSED_DATA_DIR
    lines = []

    for jsonl_file in data_dir.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                if "tlh" in pair and pair["tlh"].strip():
                    lines.append(pair["tlh"].strip())

    return "\n".join(lines)


def train_klingon_spm(
    klingon_text: str,
    output_dir: Path | None = None,
    vocab_size: int = 1000,
) -> Path:
    """Train a SentencePiece model on Klingon text.

    Args:
        klingon_text: Klingon text corpus (one sentence per line).
        output_dir: Where to save the SPM model.
        vocab_size: Target vocabulary size for Klingon subwords.

    Returns:
        Path to the trained .model file.
    """
    output_dir = output_dir or MODELS_DIR / "klingon_spm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write text to temp file for SPM training
    text_file = output_dir / "klingon_corpus.txt"
    text_file.write_text(klingon_text, encoding="utf-8")

    model_prefix = output_dir / "klingon_spm"
    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,  # Full coverage for a constructed language
        model_type="bpe",
        num_threads=4,
    )

    print(f"Trained Klingon SentencePiece model: {model_prefix}.model")
    return Path(f"{model_prefix}.model")


def extend_nllb_tokenizer(
    spm_model_path: Path,
    output_dir: Path | None = None,
) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Extend NLLB tokenizer and model with Klingon tokens.

    Args:
        spm_model_path: Path to trained Klingon SentencePiece .model file.
        output_dir: Where to save the extended model.

    Returns:
        Tuple of (extended_tokenizer, extended_model).
    """
    output_dir = output_dir or MODELS_DIR / "nllb-klingon-extended"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base NLLB model and tokenizer
    print(f"Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)

    # Load trained Klingon SPM and get its vocabulary
    klingon_spm = spm.SentencePieceProcessor()
    klingon_spm.load(str(spm_model_path))

    # Find tokens in Klingon SPM that aren't in NLLB's vocabulary
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = []
    for i in range(klingon_spm.get_piece_size()):
        token = klingon_spm.id_to_piece(i)
        if token not in existing_vocab:
            new_tokens.append(token)

    # Add the Klingon language code as a special token
    if KLINGON_CODE not in existing_vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": [KLINGON_CODE]})

    # Add new Klingon subword tokens
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"Added {len(new_tokens)} new Klingon tokens to vocabulary")

    # Resize model embeddings to match new vocabulary size
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_size = model.get_input_embeddings().weight.shape[0]

    # Initialize new embeddings from English embeddings (transfer learning)
    if new_size > old_size:
        eng_id = tokenizer.convert_tokens_to_ids(ENGLISH_CODE)
        with torch.no_grad():
            # Copy English language code embedding to Klingon language code
            tlh_id = tokenizer.convert_tokens_to_ids(KLINGON_CODE)
            model.get_input_embeddings().weight[tlh_id] = (
                model.get_input_embeddings().weight[eng_id].clone()
            )

            # Initialize other new token embeddings with mean of existing embeddings
            mean_embedding = model.get_input_embeddings().weight[:old_size].mean(dim=0)
            for i in range(old_size, new_size):
                if i != tlh_id:
                    model.get_input_embeddings().weight[i] = mean_embedding.clone()

        print(f"Initialized {new_size - old_size} new embeddings")

    # Save extended model and tokenizer
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved extended model to: {output_dir}")

    return tokenizer, model


if __name__ == "__main__":
    print("=== Klingon Tokenizer Extension Pipeline ===\n")

    print("Step 1: Collecting Klingon text...")
    text = collect_klingon_text()
    if not text:
        print("No Klingon text found. Run data download first:")
        print("  python -m klingon_translator.data.download")
    else:
        print(f"Collected {len(text.splitlines())} lines of Klingon text\n")

        print("Step 2: Training SentencePiece model...")
        spm_path = train_klingon_spm(text)

        print("\nStep 3: Extending NLLB tokenizer and model...")
        tok, mdl = extend_nllb_tokenizer(spm_path)
        print("\nDone! Extended model ready for fine-tuning.")
