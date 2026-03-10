"""Data acquisition and preprocessing for Klingon-English parallel corpus."""

import json
import random
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from klingon_translator.utils.config import (
    DEFAULT_TEST_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_VAL_SPLIT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_dirs,
)


def download_tatoeba() -> list[dict[str, str]]:
    """Download Klingon-English sentence pairs from Tatoeba via OPUS.

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    ensure_dirs()
    output_dir = RAW_DATA_DIR / "tatoeba"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "opus_read",
                "-d", "Tatoeba",
                "-s", "en",
                "-t", "tlh",
                "-p", "raw",
                "-wm", "moses",
                "-w", str(output_dir / "en.txt"), str(output_dir / "tlh.txt"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not download Tatoeba data: {e}")
        print("You may need to install opustools: pip install opustools")
        return []

    pairs = []
    en_file = output_dir / "en.txt"
    tlh_file = output_dir / "tlh.txt"
    if en_file.exists() and tlh_file.exists():
        en_lines = en_file.read_text(encoding="utf-8").strip().splitlines()
        tlh_lines = tlh_file.read_text(encoding="utf-8").strip().splitlines()
        for en, tlh in zip(en_lines, tlh_lines):
            en, tlh = en.strip(), tlh.strip()
            if en and tlh:
                pairs.append({"en": en, "tlh": tlh})

    print(f"Tatoeba: downloaded {len(pairs)} sentence pairs")
    return pairs


def parse_boqwi(xml_path: Path | None = None) -> list[dict[str, str]]:
    """Parse boQwI' Klingon dictionary data for parallel entries.

    Args:
        xml_path: Path to the boQwI' XML data file. If None, looks in raw data dir.

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    if xml_path is None:
        xml_path = RAW_DATA_DIR / "boqwi" / "KlingonAssistant" / "data" / "mem-primary.xml"

    if not xml_path.exists():
        print(f"boQwI' data not found at {xml_path}")
        print("Clone it with: git clone https://github.com/De7vID/klingon-assistant-data")
        return []

    pairs = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for entry in root.iter("entry"):
        tlh = entry.get("name", "").strip()
        en = entry.get("definition", "").strip()
        if tlh and en:
            pairs.append({"en": en, "tlh": tlh})

        # Also extract example sentences if available
        for example in entry.iter("example"):
            ex_tlh = example.get("src", "").strip()
            ex_en = example.get("dst", "").strip()
            if ex_tlh and ex_en:
                pairs.append({"en": ex_en, "tlh": ex_tlh})

    print(f"boQwI': parsed {len(pairs)} entries")
    return pairs


def load_proverbs() -> list[dict[str, str]]:
    """Load curated Klingon proverbs and common phrases.

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    proverbs_file = RAW_DATA_DIR / "proverbs.json"
    if not proverbs_file.exists():
        print(f"No proverbs file at {proverbs_file}, creating sample...")
        sample = [
            {"tlh": "Heghlu'meH QaQ jajvam.", "en": "Today is a good day to die."},
            {"tlh": "bortaS bIr jablu'DI' reH QaQqu' nay'.", "en": "Revenge is a dish best served cold."},
            {"tlh": "qaStaHvIS wa' ram loS SaD Hugh SIjlaH qetbogh loD.", "en": "A running man can slit four thousand throats in one night."},
            {"tlh": "Qu'vatlh!", "en": "Damn!"},
            {"tlh": "nuqneH", "en": "What do you want?"},
            {"tlh": "Qapla'!", "en": "Success!"},
            {"tlh": "yIDoghQo'!", "en": "Don't be silly!"},
            {"tlh": "wo' batlhvaD.", "en": "For the honor of the Empire."},
            {"tlh": "DabuQlu'DI' yISuv.", "en": "When threatened, fight."},
            {"tlh": "bIlujDI' yIchegh.", "en": "When you fail, return."},
            {"tlh": "not lay'Ha' lulIjlaHbe'bogh.", "en": "What cannot be forgotten cannot be dishonored."},
            {"tlh": "tIqDaq HoSna' tu'lu'.", "en": "Real strength is found in the heart."},
        ]
        proverbs_file.parent.mkdir(parents=True, exist_ok=True)
        proverbs_file.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")

    data = json.loads(proverbs_file.read_text(encoding="utf-8"))
    print(f"Proverbs: loaded {len(data)} entries")
    return data


def build_dataset(
    train_ratio: float = DEFAULT_TRAIN_SPLIT,
    val_ratio: float = DEFAULT_VAL_SPLIT,
    test_ratio: float = DEFAULT_TEST_SPLIT,
) -> dict[str, list[dict[str, str]]]:
    """Combine all data sources, deduplicate, and split into train/val/test.

    Returns:
        Dict with keys "train", "val", "test", each a list of {"en", "tlh"} dicts.
    """
    ensure_dirs()

    # Collect from all sources
    all_pairs = []
    all_pairs.extend(download_tatoeba())
    all_pairs.extend(parse_boqwi())
    all_pairs.extend(load_proverbs())

    # Deduplicate by (en, tlh) tuple
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = (pair["en"].lower(), pair["tlh"].lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"Total unique pairs: {len(unique_pairs)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(unique_pairs)

    n = len(unique_pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": unique_pairs[:train_end],
        "val": unique_pairs[train_end:val_end],
        "test": unique_pairs[val_end:],
    }

    # Save each split as JSON Lines
    for split_name, data in splits.items():
        out_path = PROCESSED_DATA_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in data:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Saved {split_name}: {len(data)} pairs -> {out_path}")

    return splits


if __name__ == "__main__":
    print("Building Klingon-English dataset...")
    splits = build_dataset()
    print(f"\nDone! Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
