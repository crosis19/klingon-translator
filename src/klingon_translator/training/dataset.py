"""Bidirectional translation dataset with pre-tokenization."""

import random
import time

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from klingon_translator.utils.config import ENGLISH_CODE, KLINGON_CODE


class BilingualDataset(Dataset):
    """Bidirectional translation dataset with pre-tokenization.

    For each pair, creates two training examples:
    - English -> Klingon
    - Klingon -> English

    All tokenization happens once in __init__() and results are cached
    in memory. __getitem__() is a simple list lookup.

    Args:
        pairs: List of {"en": ..., "tlh": ...} dicts.
        tokenizer: The extended NLLB tokenizer.
        max_length: Maximum token sequence length.
        shuffle: Whether to shuffle the cached examples.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        pairs: list[dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.cached: list[dict] = []

        print(f"  Pre-tokenizing {len(pairs):,} pairs (2 directions each)...")
        t0 = time.time()

        eng_lang_id = tokenizer.convert_tokens_to_ids(ENGLISH_CODE)
        tlh_lang_id = tokenizer.convert_tokens_to_ids(KLINGON_CODE)

        for p in pairs:
            # en -> tlh
            tokenizer.src_lang = ENGLISH_CODE
            src = tokenizer(
                p["en"], truncation=True, max_length=max_length
            )
            tokenizer.src_lang = KLINGON_CODE
            tgt = tokenizer(
                p["tlh"], truncation=True, max_length=max_length
            )
            self.cached.append({
                "input_ids": src["input_ids"],
                "attention_mask": src["attention_mask"],
                "labels": [tlh_lang_id] + tgt["input_ids"],
            })

            # tlh -> en
            tokenizer.src_lang = KLINGON_CODE
            src = tokenizer(
                p["tlh"], truncation=True, max_length=max_length
            )
            tokenizer.src_lang = ENGLISH_CODE
            tgt = tokenizer(
                p["en"], truncation=True, max_length=max_length
            )
            self.cached.append({
                "input_ids": src["input_ids"],
                "attention_mask": src["attention_mask"],
                "labels": [eng_lang_id] + tgt["input_ids"],
            })

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.cached)

        elapsed = time.time() - t0
        print(f"  Done: {len(self.cached):,} examples in {elapsed:.1f}s")

    def __len__(self) -> int:
        return len(self.cached)

    def __getitem__(self, idx: int) -> dict:
        return self.cached[idx]
