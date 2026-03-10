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
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

from klingon_translator.utils.config import (
    BASE_MODEL_ID,
    ENGLISH_CODE,
    KLINGON_CODE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def collect_klingon_text(data_dir: Path | None = None) -> str:
    """Collect all available Klingon text for tokenizer training.

    Gathers Klingon text from processed JSONL files. When using the default
    data directory, also includes raw cached data for maximum coverage.

    Args:
        data_dir: Directory containing processed .jsonl files.
            If None, uses the default PROCESSED_DATA_DIR and also reads raw data.

    Returns:
        Concatenated Klingon text, one sentence per line.
    """
    use_defaults = data_dir is None
    data_dir = data_dir or PROCESSED_DATA_DIR
    lines = []

    # Collect from processed JSONL files (train/val/test splits)
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                if "tlh" in pair and pair["tlh"].strip():
                    lines.append(pair["tlh"].strip())

    # Also collect from raw cached data for maximum coverage (default dir only)
    raw_dir = RAW_DATA_DIR
    if use_defaults and raw_dir.exists():
        # Tatoeba cache
        tatoeba_cache = raw_dir / "tatoeba_pairs.json"
        if tatoeba_cache.exists():
            pairs = json.loads(tatoeba_cache.read_text(encoding="utf-8"))
            for pair in pairs:
                if pair.get("tlh", "").strip():
                    lines.append(pair["tlh"].strip())

        # Proverbs
        proverbs_file = raw_dir / "proverbs.json"
        if proverbs_file.exists():
            proverbs = json.loads(proverbs_file.read_text(encoding="utf-8"))
            for p in proverbs:
                if p.get("tlh", "").strip():
                    lines.append(p["tlh"].strip())

    # Deduplicate while preserving order
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return "\n".join(unique_lines)


def train_klingon_spm(
    klingon_text: str,
    output_dir: Path | None = None,
    vocab_size: int = 1000,
) -> Path:
    """Train a SentencePiece model on Klingon text.

    Uses BPE with full character coverage since Klingon is a constructed
    language with a small, well-defined character set.

    Args:
        klingon_text: Klingon text corpus (one sentence per line).
        output_dir: Where to save the SPM model.
        vocab_size: Target vocabulary size for Klingon subwords.

    Returns:
        Path to the trained .model file.
    """
    output_dir = output_dir or MODELS_DIR / "klingon_spm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write text to file for SPM training
    text_file = output_dir / "klingon_corpus.txt"
    text_file.write_text(klingon_text, encoding="utf-8")

    num_lines = len(klingon_text.splitlines())
    print(f"Training SentencePiece on {num_lines} Klingon sentences...")

    # Cap vocab_size for small corpora. SentencePiece needs enough data
    # to create the requested number of subword merges.
    # Heuristic: unique chars + some merges, capped to available data
    unique_chars = len(set(klingon_text.replace("\n", "")))
    max_feasible = unique_chars + num_lines  # conservative upper bound
    effective_vocab = min(vocab_size, max(32, max_feasible))
    if effective_vocab != vocab_size:
        print(f"  Adjusted vocab_size: {vocab_size} -> {effective_vocab} (small corpus)")

    model_prefix = output_dir / "klingon_spm"
    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(model_prefix),
        vocab_size=effective_vocab,
        character_coverage=1.0,  # Full coverage for a constructed language
        model_type="bpe",
        num_threads=4,
        pad_id=3,  # Match NLLB convention
        hard_vocab_limit=False,  # Allow smaller vocab if not enough data
    )

    model_path = Path(f"{model_prefix}.model")
    # Verify the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    print(f"Trained Klingon SPM: {sp.get_piece_size()} tokens -> {model_path}")

    return model_path


def extend_nllb_tokenizer(
    spm_model_path: Path,
    output_dir: Path | None = None,
) -> tuple:
    """Extend NLLB tokenizer and model with Klingon tokens.

    Steps:
    1. Load the base NLLB-200 model and tokenizer
    2. Load the trained Klingon SentencePiece vocabulary
    3. Add new Klingon subword tokens to the NLLB tokenizer
    4. Register tlh_Latn as a new language code
    5. Resize model embeddings and initialize new tokens from English
    6. Save the extended model

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
    print("  (This downloads ~2.3 GB on first run)")
    tokenizer = NllbTokenizerFast.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)

    print(f"  Base vocab size: {len(tokenizer)}")
    print(f"  Model embedding size: {model.get_input_embeddings().weight.shape}")

    # Load trained Klingon SPM and get its vocabulary
    klingon_spm = spm.SentencePieceProcessor()
    klingon_spm.load(str(spm_model_path))

    # Find tokens in Klingon SPM that aren't already in NLLB's vocabulary
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = []
    for i in range(klingon_spm.get_piece_size()):
        token = klingon_spm.id_to_piece(i)
        # Skip SPM control tokens (<unk>, <s>, </s>, <pad>)
        if token.startswith("<") and token.endswith(">"):
            continue
        if token not in existing_vocab:
            new_tokens.append(token)

    print(f"  New Klingon tokens to add: {len(new_tokens)}")

    # Register the Klingon language code as a special token
    # NLLB uses language codes like "eng_Latn" as special tokens for
    # controlling translation direction via forced_bos_token_id
    lang_code_added = False
    if KLINGON_CODE not in existing_vocab:
        current_special = tokenizer.special_tokens_map.get("additional_special_tokens", [])
        if KLINGON_CODE not in current_special:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": current_special + [KLINGON_CODE]}
            )
            lang_code_added = True
            print(f"  Registered language code: {KLINGON_CODE}")

    # Add new Klingon subword tokens
    num_added = 0
    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"  Added {num_added} new tokens to tokenizer")

    if num_added == 0 and not lang_code_added:
        print("  No new tokens needed - vocabulary already covers Klingon text")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        return tokenizer, model

    # Resize model embeddings to match new vocabulary size
    old_emb_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_emb_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Resized embeddings: {old_emb_size} -> {new_emb_size}")

    # Initialize new embeddings via transfer learning from English
    if new_emb_size > old_emb_size:
        eng_id = tokenizer.convert_tokens_to_ids(ENGLISH_CODE)
        tlh_id = tokenizer.convert_tokens_to_ids(KLINGON_CODE)

        with torch.no_grad():
            input_emb = model.get_input_embeddings().weight
            output_emb = model.get_output_embeddings().weight

            # Compute mean embedding from all original tokens for initializing
            # new subword tokens (a reasonable prior)
            mean_input_emb = input_emb[:old_emb_size].mean(dim=0)
            mean_output_emb = output_emb[:old_emb_size].mean(dim=0)

            for i in range(old_emb_size, new_emb_size):
                if i == tlh_id:
                    # Initialize Klingon language code from English language code
                    # (closest semantic proxy for a new language)
                    input_emb[i] = input_emb[eng_id].clone()
                    output_emb[i] = output_emb[eng_id].clone()
                else:
                    # Initialize new subword tokens with mean embedding
                    input_emb[i] = mean_input_emb.clone()
                    output_emb[i] = mean_output_emb.clone()

        print(f"  Initialized {new_emb_size - old_emb_size} new embeddings")
        print(f"    Klingon lang code (id={tlh_id}) <- English (id={eng_id})")
        print(f"    Other tokens <- mean embedding")

    # Save extended model and tokenizer
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"\nSaved extended model to: {output_dir}")

    # Quick verification
    test_text = "Qapla'"
    encoded = tokenizer(test_text)
    print(f"\nVerification: '{test_text}' -> token IDs: {encoded['input_ids']}")
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    print(f"  Decoded back: '{decoded}'")

    return tokenizer, model


def run_pipeline(vocab_size: int = 1000) -> tuple:
    """Run the complete tokenizer extension pipeline.

    Convenience function that chains collect -> train -> extend.

    Args:
        vocab_size: Target Klingon vocabulary size.

    Returns:
        Tuple of (extended_tokenizer, extended_model).
    """
    print("=" * 60)
    print("Klingon Tokenizer Extension Pipeline")
    print("=" * 60)

    print("\nStep 1/3: Collecting Klingon text...")
    text = collect_klingon_text()
    if not text:
        raise RuntimeError(
            "No Klingon text found. Run data download first:\n"
            "  python -m klingon_translator.data.download"
        )
    num_lines = len(text.splitlines())
    print(f"  Collected {num_lines} unique Klingon sentences")

    print("\nStep 2/3: Training SentencePiece model...")
    spm_path = train_klingon_spm(text, vocab_size=vocab_size)

    print("\nStep 3/3: Extending NLLB tokenizer and model...")
    tokenizer, model = extend_nllb_tokenizer(spm_path)

    print("\n" + "=" * 60)
    print("Done! Extended model ready for fine-tuning.")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    return tokenizer, model


if __name__ == "__main__":
    run_pipeline()
