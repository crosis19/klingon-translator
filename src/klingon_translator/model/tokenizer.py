"""Klingon tokenizer extension for NLLB-200.

Since NLLB doesn't include Klingon, we need to:
1. Train a SentencePiece model on available Klingon text
2. Merge new Klingon subword tokens into NLLB's tokenizer
3. Register the tlh_Latn language code
4. Initialize new token embeddings via base-tokenizer decomposition (transfer learning)
"""

import json
from pathlib import Path

import sentencepiece as spm
import torch
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

from klingon_translator.utils.config import (
    BASE_MODEL_ID,
    ENGLISH_CODE,
    KLINGON_CODE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def collect_klingon_text(
    data_dir: Path | None = None,
    raw_data_dir: Path | None = None,
) -> str:
    """Collect all available Klingon text for tokenizer training.

    Gathers Klingon text from processed JSONL files. When using the default
    data directory, also includes raw cached data for maximum coverage.

    Args:
        data_dir: Directory containing processed .jsonl files.
            If None, uses the default PROCESSED_DATA_DIR and also reads raw data.
        raw_data_dir: Directory containing raw data files (OPUS, paq'batlh, etc.).
            If None, uses the default RAW_DATA_DIR when data_dir is also None.

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

    # Also collect from raw cached data for maximum coverage
    raw_dir = raw_data_dir or RAW_DATA_DIR
    use_raw = use_defaults or raw_data_dir is not None
    if use_raw and raw_dir.exists():
        # Tatoeba API cache
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

        # OPUS Moses Klingon file (one sentence per line)
        opus_tlh = raw_dir / "opus" / "tatoeba" / "Tatoeba.en-tlh.tlh"
        if opus_tlh.exists():
            for line in opus_tlh.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    lines.append(line)

        # paq'batlh pairs
        paqbatlh_cache = raw_dir / "paqbatlh_pairs.json"
        if paqbatlh_cache.exists():
            paq_pairs = json.loads(
                paqbatlh_cache.read_text(encoding="utf-8")
            )
            for pair in paq_pairs:
                if pair.get("tlh", "").strip():
                    lines.append(pair["tlh"].strip())

        # boQwI' monolingual Klingon data (dictionary entries,
        # examples, notes — maximizes SPM coverage)
        boqwi_mono = raw_dir / "boqwi_monolingual.txt"
        if boqwi_mono.exists():
            for line in boqwi_mono.read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if line:
                    lines.append(line)

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
    vocab_size: int | None = None,
) -> Path:
    """Train a SentencePiece model on Klingon text.

    Uses BPE with full character coverage since Klingon is a constructed
    language with a small, well-defined character set.

    Args:
        klingon_text: Klingon text corpus (one sentence per line).
        output_dir: Where to save the SPM model.
        vocab_size: Target vocabulary size for Klingon subwords.
            If None, auto-scales based on corpus size (1000-4000).

    Returns:
        Path to the trained .model file.
    """
    output_dir = output_dir or MODELS_DIR / "klingon_spm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write text to file for SPM training
    text_file = output_dir / "klingon_corpus.txt"
    text_file.write_text(klingon_text, encoding="utf-8")

    num_lines = len(klingon_text.splitlines())

    # Auto-scale vocab_size based on corpus size if not specified
    if vocab_size is None:
        num_unique = len(set(klingon_text.splitlines()))
        vocab_size = min(4000, max(1000, num_unique // 5))
        print(f"  Auto vocab_size: {vocab_size} (from {num_unique} unique sentences)")

    print(f"Training SentencePiece on {num_lines} Klingon sentences...")

    # Cap vocab_size for small corpora. SentencePiece needs enough data
    # to create the requested number of subword merges.
    # byte_fallback=True reserves 256 byte tokens + 4 control + 1 user symbol
    BYTE_FALLBACK_OVERHEAD = 261  # 256 bytes + <unk> <s> </s> <pad> + apostrophe
    unique_chars = len(set(klingon_text.replace("\n", "")))
    min_required = BYTE_FALLBACK_OVERHEAD + unique_chars
    max_feasible = unique_chars + num_lines  # conservative upper bound
    effective_vocab = min(vocab_size, max(min_required, max_feasible))
    if effective_vocab != vocab_size:
        print(
            f"  Adjusted vocab_size: {vocab_size} -> {effective_vocab}"
            " (small corpus)"
        )

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
        byte_fallback=True,  # Unknown chars -> bytes instead of <unk>
        split_digits=True,  # Digits handled individually
        user_defined_symbols=["'"],  # Apostrophe is atomic (central to Klingon)
        normalization_rule_name="identity",  # No normalization (preserve Klingon orthography)
        allow_whitespace_only_pieces=False,  # Prevent whitespace-only tokens
        remove_extra_whitespaces=True,  # Clean up whitespace in training data
        add_dummy_prefix=True,  # Match NLLB's ▁ prefix convention
    )

    model_path = Path(f"{model_prefix}.model")
    # Verify the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    print(f"Trained Klingon SPM: {sp.get_piece_size()} tokens -> {model_path}")

    return model_path


def _add_tokens_to_bpe(
    tokenizer: NllbTokenizerFast,
    new_tokens: list[str],
) -> int:
    """Add tokens directly to the BPE model's vocab and merge rules.

    Unlike ``tokenizer.add_tokens()``, this injects tokens into the
    underlying BPE model so they participate in normal BPE merging.
    ``add_tokens()`` creates "added tokens" that are matched on the
    *raw* text before the Metaspace pre-tokenizer runs, which breaks
    word-internal subword boundaries and causes spurious spaces
    (e.g. ``Heghlu'meH`` → ``"Hegh lu'meH"``).

    By adding directly to the BPE vocab + merges, the Metaspace
    pre-tokenizer handles ``▁`` word boundaries correctly and
    tokenization round-trips are preserved.

    Args:
        tokenizer: The NLLB fast tokenizer to modify in-place.
        new_tokens: Tokens from the Klingon SPM that are not in
            the base NLLB vocabulary.

    Returns:
        Number of tokens actually added.
    """
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    vocab = tok_json["model"]["vocab"]
    merges = tok_json["model"]["merges"]
    existing_merges = set(tuple(m) for m in merges)

    vocab_set = set(vocab.keys())
    next_id = max(vocab.values()) + 1
    num_added = 0

    # Add each new token to the BPE vocabulary
    for token in new_tokens:
        if token not in vocab_set:
            vocab[token] = next_id
            next_id += 1
            vocab_set.add(token)
            num_added += 1

    # Add merge rules so BPE can compose these tokens.
    # For each multi-char token, find a split where both halves
    # already exist in the vocabulary.
    new_merges = []
    for token in new_tokens:
        if len(token) <= 1:
            continue
        for split_pos in range(1, len(token)):
            left = token[:split_pos]
            right = token[split_pos:]
            if left in vocab_set and right in vocab_set:
                pair = (left, right)
                if pair not in existing_merges:
                    new_merges.append([left, right])
                    existing_merges.add(pair)
                break

    merges.extend(new_merges)

    # Rebuild the backend tokenizer from modified JSON
    tok_json["model"]["vocab"] = vocab
    tok_json["model"]["merges"] = merges
    tokenizer._tokenizer = HFTokenizer.from_str(
        json.dumps(tok_json)
    )

    return num_added


def extend_nllb_tokenizer(
    spm_model_path: Path,
    output_dir: Path | None = None,
) -> tuple:
    """Extend NLLB tokenizer and model with Klingon tokens.

    Steps:
    1. Load the base NLLB-200 model and tokenizer
    2. Load the trained Klingon SentencePiece vocabulary
    3. Pre-compute embeddings for new tokens via base-tokenizer
       decomposition
    4. Add Klingon subword tokens to the BPE model (not as "added
       tokens") so that Metaspace word-boundary handling is preserved
    5. Register the tlh_Latn language code as a special token
    6. Resize model embeddings and initialize from pre-computed values
    7. Save the extended model

    Args:
        spm_model_path: Path to trained Klingon SentencePiece .model
            file.
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
    emb_shape = model.get_input_embeddings().weight.shape
    print(f"  Model embedding size: {emb_shape}")

    # Load trained Klingon SPM and get its vocabulary
    klingon_spm = spm.SentencePieceProcessor()
    klingon_spm.load(str(spm_model_path))

    # Find tokens in Klingon SPM not already in NLLB
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = []
    for i in range(klingon_spm.get_piece_size()):
        token = klingon_spm.id_to_piece(i)
        if token.startswith("<") and token.endswith(">"):
            continue
        if token not in existing_vocab:
            new_tokens.append(token)

    print(f"  New Klingon tokens to add: {len(new_tokens)}")

    # ── Pre-compute embeddings BEFORE modifying tokenizer ──────
    # For each new token, tokenize it with the original NLLB
    # tokenizer (which splits it into existing subwords) and
    # average those embeddings for a meaningful initialization.
    old_emb_size = model.get_input_embeddings().weight.shape[0]
    token_init_map = {}

    with torch.no_grad():
        input_emb = model.get_input_embeddings().weight
        output_emb = model.get_output_embeddings().weight
        mean_input = input_emb[:old_emb_size].mean(dim=0)
        mean_output = output_emb[:old_emb_size].mean(dim=0)

        for token in new_tokens:
            base_ids = tokenizer.encode(
                token, add_special_tokens=False
            )
            if base_ids:
                avg_in = input_emb[base_ids].mean(dim=0)
                avg_out = output_emb[base_ids].mean(dim=0)
            else:
                avg_in = mean_input
                avg_out = mean_output
            token_init_map[token] = (
                avg_in.clone(),
                avg_out.clone(),
            )

    print(
        f"  Pre-computed embeddings for "
        f"{len(token_init_map)} new tokens"
    )

    # ── Add Klingon tokens to BPE model ────────────────────────
    # Tokens are injected into the BPE vocab + merge rules rather
    # than via add_tokens() to preserve Metaspace word boundaries.
    num_added = _add_tokens_to_bpe(tokenizer, new_tokens)
    print(f"  Added {num_added} tokens to BPE model")

    # ── Register Klingon language code ─────────────────────────
    lang_code_added = False
    if KLINGON_CODE not in existing_vocab:
        current_special = tokenizer.special_tokens_map.get(
            "additional_special_tokens", []
        )
        if KLINGON_CODE not in current_special:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": (
                        current_special + [KLINGON_CODE]
                    )
                }
            )
            lang_code_added = True
            print(f"  Registered language code: {KLINGON_CODE}")

    if num_added == 0 and not lang_code_added:
        print("  No new tokens needed")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        return tokenizer, model

    # ── Resize embeddings and initialize ───────────────────────
    model.resize_token_embeddings(len(tokenizer))
    new_emb_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Resized embeddings: {old_emb_size} -> {new_emb_size}")

    if new_emb_size > old_emb_size:
        eng_id = tokenizer.convert_tokens_to_ids(ENGLISH_CODE)
        tlh_id = tokenizer.convert_tokens_to_ids(KLINGON_CODE)

        with torch.no_grad():
            input_emb = model.get_input_embeddings().weight
            output_emb = model.get_output_embeddings().weight

            for i in range(old_emb_size, new_emb_size):
                if i == tlh_id:
                    input_emb[i] = input_emb[eng_id].clone()
                    output_emb[i] = output_emb[eng_id].clone()
                else:
                    tok_str = tokenizer.convert_ids_to_tokens(i)
                    if tok_str in token_init_map:
                        ie, oe = token_init_map[tok_str]
                        input_emb[i] = ie
                        output_emb[i] = oe
                    else:
                        input_emb[i] = mean_input.clone()
                        output_emb[i] = mean_output.clone()

        n_new = new_emb_size - old_emb_size
        print(f"  Initialized {n_new} new embeddings")
        print(
            f"    tlh_Latn (id={tlh_id}) <- "
            f"eng_Latn (id={eng_id})"
        )
        print("    Others <- base-tokenizer decomposition")

    # ── Roundtrip verification ─────────────────────────────────
    verify_phrases = [
        "Heghlu'meH QaQ jajvam.",
        "Qapla'!",
        "nuqneH?",
        "tlhIngan maH!",
    ]
    print("\n  Roundtrip verification:")
    all_ok = True
    for phrase in verify_phrases:
        ids = tokenizer(phrase, add_special_tokens=False)[
            "input_ids"
        ]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        ok = phrase == decoded.strip()
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] {phrase}")
        if not ok:
            print(f"           -> {decoded}")
    if all_ok:
        print("    All roundtrips passed!")

    # Save extended model and tokenizer
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"\nSaved extended model to: {output_dir}")

    # Quality report
    klingon_spm_token_count = sum(
        1
        for i in range(klingon_spm.get_piece_size())
        if not (
            klingon_spm.id_to_piece(i).startswith("<")
            and klingon_spm.id_to_piece(i).endswith(">")
        )
    )
    corpus_file = spm_model_path.parent / "klingon_corpus.txt"
    if corpus_file.exists():
        klingon_text = corpus_file.read_text(encoding="utf-8")
        report_tokenizer_quality(
            tokenizer,
            klingon_text,
            num_new_tokens=num_added,
            klingon_spm_vocab_size=klingon_spm_token_count,
        )

    return tokenizer, model


def report_tokenizer_quality(
    tokenizer: NllbTokenizerFast,
    klingon_text: str,
    num_new_tokens: int | None = None,
    klingon_spm_vocab_size: int | None = None,
) -> dict:
    """Compute and print tokenizer quality metrics on Klingon text.

    Metrics:
        - Fertility: avg subword tokens per whitespace word (lower = better)
        - Coverage: % of Klingon SPM vocab that was new vs already in NLLB
        - Sample tokenizations of representative Klingon phrases

    Args:
        tokenizer: The extended NLLB tokenizer.
        klingon_text: Klingon corpus text (one sentence per line).
        num_new_tokens: Number of new tokens added (for coverage calc).
        klingon_spm_vocab_size: Total Klingon SPM vocab size (non-control tokens).

    Returns:
        Dict with keys "fertility", "coverage_pct", "sample_tokenizations".
    """
    print("\n" + "=" * 60)
    print("Tokenizer Quality Report")
    print("=" * 60)

    # ── Fertility ────────────────────────────────────────────────────
    sentences = [s.strip() for s in klingon_text.splitlines() if s.strip()]
    total_words = 0
    total_tokens = 0
    for sent in sentences:
        words = sent.split()
        total_words += len(words)
        token_ids = tokenizer.encode(sent, add_special_tokens=False)
        total_tokens += len(token_ids)

    fertility = total_tokens / total_words if total_words > 0 else 0.0
    print(f"\nFertility: {fertility:.2f} tokens/word")
    print(
        f"  ({total_tokens:,} tokens for {total_words:,} words "
        f"across {len(sentences):,} sentences)"
    )

    # ── Coverage ─────────────────────────────────────────────────────
    coverage_pct = None
    if num_new_tokens is not None and klingon_spm_vocab_size is not None:
        already_in_nllb = klingon_spm_vocab_size - num_new_tokens
        coverage_pct = (num_new_tokens / klingon_spm_vocab_size) * 100
        print("\nVocab coverage:")
        print(f"  Klingon SPM vocab: {klingon_spm_vocab_size}")
        print(f"  New tokens added:  {num_new_tokens} ({coverage_pct:.1f}%)")
        print(
            f"  Already in NLLB:   {already_in_nllb} "
            f"({100 - coverage_pct:.1f}%)"
        )

    # ── Sample tokenizations ─────────────────────────────────────────
    sample_phrases = [
        "Qapla'",
        "tlhIngan maH",
        "nuqneH",
        "bortaS bIr jablu'DI' reH QaQqu' nay'",
        "Heghlu'meH QaQ jajvam",
        "DabuQlu'DI' yISuv",
    ]
    print("\nSample tokenizations:")
    for phrase in sample_phrases:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {phrase}")
        print(f"    -> {tokens} ({len(tokens)} tokens)")

    return {
        "fertility": fertility,
        "coverage_pct": coverage_pct,
        "sample_tokenizations": sample_phrases,
    }


def run_pipeline(vocab_size: int | None = None) -> tuple:
    """Run the complete tokenizer extension pipeline.

    Convenience function that chains collect -> train -> extend.

    Args:
        vocab_size: Target Klingon vocabulary size.
            If None, auto-scales based on corpus size (1000-4000).

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
