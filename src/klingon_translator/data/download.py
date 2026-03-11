"""Data acquisition and preprocessing for Klingon-English parallel corpus.

Data sources:
    1. Tatoeba API: Community-translated sentence pairs via tatoeba.org
    2. OPUS Tatoeba: Larger parallel corpus from OPUS project (Moses format)
    3. boQwI' dictionary: YAML entries from De7vID/klingon-assistant-data
    4. paq'batlh: Klingon epic poem, bilingual edition (CC BY-NC-SA 4.0)
    5. Curated proverbs: Hand-collected Klingon sayings with translations
"""

import json
import random
import re
import ssl
import time
import urllib.request
from pathlib import Path

import yaml

from klingon_translator.utils.config import (
    DEFAULT_TEST_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_VAL_SPLIT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_dirs,
)

# SSL context that skips verification (Tatoeba has cert issues)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def download_tatoeba(max_pages: int = 100) -> list[dict[str, str]]:
    """Download Klingon-English sentence pairs from Tatoeba website API.

    Uses the tatoeba.org search API to find Klingon sentences that have
    English translations. Paginates through all available results.

    Args:
        max_pages: Maximum number of pages to fetch (10 results per page).

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    ensure_dirs()
    cache_file = RAW_DATA_DIR / "tatoeba_pairs.json"

    # Use cached data if available
    if cache_file.exists():
        pairs = json.loads(cache_file.read_text(encoding="utf-8"))
        print(f"Tatoeba API: loaded {len(pairs)} cached pairs")
        return pairs

    print("Tatoeba API: downloading from API (this may take a few minutes)...")
    pairs = []
    base_url = "https://tatoeba.org/en/api_v0/search?from=tlh&to=eng&query=&page={}"

    for page in range(1, max_pages + 1):
        url = base_url.format(page)
        req = urllib.request.Request(
            url, headers={"User-Agent": "KlingonTranslator/0.1"}
        )

        try:
            resp = urllib.request.urlopen(req, context=_SSL_CTX, timeout=30)
            data = json.loads(resp.read())
        except Exception as e:
            print(f"  Warning: page {page} failed: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for result in results:
            tlh_text = result.get("text", "").strip()
            # Translations are nested: [[direct], [indirect]]
            translations = result.get("translations", [])
            for group in translations:
                for trans in group:
                    if trans.get("lang") == "eng":
                        en_text = trans.get("text", "").strip()
                        if tlh_text and en_text:
                            pairs.append({"en": en_text, "tlh": tlh_text})

        paging = data.get("paging", {}).get("Sentences", {})
        has_next = paging.get("nextPage", False)

        if page % 10 == 0:
            print(f"  Page {page}: {len(pairs)} pairs so far...")

        if not has_next:
            break

        # Be polite to the API
        time.sleep(0.5)

    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Tatoeba API: downloaded {len(pairs)} sentence pairs")
    return pairs


def load_opus_tatoeba() -> list[dict[str, str]]:
    """Load Klingon-English pairs from OPUS Tatoeba corpus (Moses format).

    Expects parallel files at:
        data/raw/opus/tatoeba/Tatoeba.en-tlh.en
        data/raw/opus/tatoeba/Tatoeba.en-tlh.tlh

    These can be downloaded from: https://opus.nlpl.eu/Tatoeba/en&tlh/v2023-04-12/moses
    The OPUS corpus contains a larger set of Tatoeba pairs than the API.

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    opus_dir = RAW_DATA_DIR / "opus" / "tatoeba"
    en_file = opus_dir / "Tatoeba.en-tlh.en"
    tlh_file = opus_dir / "Tatoeba.en-tlh.tlh"

    if not en_file.exists() or not tlh_file.exists():
        print(f"OPUS Tatoeba: files not found at {opus_dir}")
        print("  Download from: https://opus.nlpl.eu/Tatoeba/en&tlh/v2023-04-12/moses")
        return []

    en_lines = en_file.read_text(encoding="utf-8").strip().splitlines()
    tlh_lines = tlh_file.read_text(encoding="utf-8").strip().splitlines()

    if len(en_lines) != len(tlh_lines):
        print(f"OPUS Tatoeba: line count mismatch ({len(en_lines)} en vs {len(tlh_lines)} tlh)")
        # Use the shorter count
        count = min(len(en_lines), len(tlh_lines))
        en_lines = en_lines[:count]
        tlh_lines = tlh_lines[:count]

    pairs = []
    skipped = 0
    for en, tlh in zip(en_lines, tlh_lines):
        en = en.strip()
        tlh = tlh.strip()
        if en and tlh:
            # Sanity check: skip if both sides are identical (mislabeled data)
            if en.lower() == tlh.lower():
                skipped += 1
                continue
            pairs.append({"en": en, "tlh": tlh})

    if skipped:
        print(f"  (skipped {skipped} identical-on-both-sides entries)")
    print(f"OPUS Tatoeba: loaded {len(pairs)} pairs")
    return pairs


def load_paqbatlh() -> list[dict[str, str]]:
    """Load English-Klingon pairs from paq'batlh (Klingon epic poem).

    The paq'batlh is a bilingual edition of the Klingon Book of Honor,
    published under CC BY-NC-SA 4.0 license. Pairs are extracted at the
    verse-line level where alignment is perfect, and at the canto level
    for sections with formatting irregularities.

    Expects pre-parsed data at:
        data/raw/paqbatlh_pairs.json

    To regenerate, run the standalone parser script with pymupdf:
        python _parse_paqbatlh.py

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    pairs_file = RAW_DATA_DIR / "paqbatlh_pairs.json"

    if not pairs_file.exists():
        print(f"paq'batlh: pre-parsed file not found at {pairs_file}")
        print("  Run the parser: python _parse_paqbatlh.py")
        return []

    raw_pairs = json.loads(pairs_file.read_text(encoding="utf-8"))

    # Convert to standard format and add source tag
    pairs = []
    for p in raw_pairs:
        en = p.get("en", "").strip()
        tlh = p.get("tlh", "").strip()
        if en and tlh:
            pairs.append({"en": en, "tlh": tlh})

    print(f"paq'batlh: loaded {len(pairs)} pairs")
    return pairs


def parse_boqwi(data_dir: Path | None = None) -> list[dict[str, str]]:
    """Parse boQwI\' Klingon dictionary YAML data for parallel entries.

    Extracts:
        - Word entries: Klingon word -> English definition
        - Sentence entries: Klingon sentence -> English translation
        - Example sentences embedded in entries

    Args:
        data_dir: Path to the klingon-assistant-data-main/entries/ directory.

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "klingon-assistant-data-main" / "entries"

    if not data_dir.exists():
        print(f"boQwI\' data not found at {data_dir}")
        print("Download from: https://github.com/De7vID/klingon-assistant-data")
        return []

    pairs = []
    skipped = 0

    for yaml_file in sorted(data_dir.rglob("*.yaml")):
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            skipped += 1
            continue

        if not data:
            continue

        # Handle files with single entry or multiple entries
        entries = []
        if "entry" in data:
            entries.append(data["entry"])
        elif "entries" in data:
            entries.extend(data["entries"])

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            tlh_name = entry.get("entry_name", "").strip()

            # Get English definition
            definition = entry.get("definition", "")
            if isinstance(definition, dict):
                en_text = definition.get("text", "").strip()
            elif isinstance(definition, str):
                en_text = definition.strip()
            else:
                en_text = ""

            # Skip template entries (contain "...")
            if "..." in tlh_name or not tlh_name or not en_text:
                continue

            # Add pair
            pairs.append({"en": en_text, "tlh": tlh_name})

            # Extract example sentences if available
            examples_text = entry.get("examples", "")
            if examples_text and isinstance(examples_text, str):
                # Examples format: {klingon:sen:nolink} "English translation"
                pattern = r'\{(.+?)(?::sen)?(?::nolink)?\}\s*["\u201c]([^"\u201d]+)["\u201d]'
                for match in re.finditer(pattern, examples_text):
                    ex_tlh = match.group(1).strip()
                    ex_en = match.group(2).strip()
                    # Clean up cross-references
                    ex_tlh = re.sub(r":[a-z_:]+$", "", ex_tlh)
                    if ex_tlh and ex_en and "..." not in ex_tlh:
                        pairs.append({"en": ex_en, "tlh": ex_tlh})

    if skipped:
        print(f"  (skipped {skipped} unreadable files)")
    print(f"boQwI\': parsed {len(pairs)} entries")
    return pairs


def load_proverbs() -> list[dict[str, str]]:
    """Load curated Klingon proverbs and common phrases.

    Creates a seed file with well-known proverbs if none exists.

    Returns:
        List of {"en": ..., "tlh": ...} dicts.
    """
    proverbs_file = RAW_DATA_DIR / "proverbs.json"
    if not proverbs_file.exists():
        print("Creating proverbs seed file...")
        sample = [
            {"tlh": "Heghlu\'meH QaQ jajvam.", "en": "Today is a good day to die."},
            {"tlh": "bortaS bIr jablu\'DI\' reH QaQqu\' nay\'.", "en": "Revenge is a dish best served cold."},
            {"tlh": "Qu\'vatlh!", "en": "Damn!"},
            {"tlh": "nuqneH.", "en": "What do you want?"},
            {"tlh": "Qapla\'!", "en": "Success!"},
            {"tlh": "yIDoghQo\'!", "en": "Don\'t be silly!"},
            {"tlh": "wo\' batlhvaD.", "en": "For the honor of the Empire."},
            {"tlh": "DabuQlu\'DI\' yISuv.", "en": "When threatened, fight."},
            {"tlh": "bIlujDI\' yIchegh.", "en": "When you fail, return."},
            {"tlh": "tIqDaq HoSna\' tu\'lu\'.", "en": "Real strength is found in the heart."},
            {"tlh": "meQtaHbogh qachDaq Suv qoH neH.", "en": "Only a fool fights in a burning house."},
            {"tlh": "taHjaj wo\'.", "en": "May the Empire endure."},
            {"tlh": "tlhIngan maH!", "en": "We are Klingons!"},
            {"tlh": "HIja\'.", "en": "Yes."},
            {"tlh": "ghobe\'.", "en": "No."},
            {"tlh": "nuqDaq \'oH puchpa\'\'e\'?", "en": "Where is the bathroom?"},
            {"tlh": "jIyajbe\'.", "en": "I don\'t understand."},
            {"tlh": "maj.", "en": "Good."},
            {"tlh": "qatlho\'.", "en": "Thank you."},
            {"tlh": "yIghoS!", "en": "Come here!"},
            {"tlh": "qoSlIj DatIvjaj.", "en": "Happy birthday."},
        ]
        proverbs_file.parent.mkdir(parents=True, exist_ok=True)
        proverbs_file.write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    data = json.loads(proverbs_file.read_text(encoding="utf-8"))
    print(f"Proverbs: loaded {len(data)} entries")
    return data


def build_dataset(
    train_ratio: float = DEFAULT_TRAIN_SPLIT,
    val_ratio: float = DEFAULT_VAL_SPLIT,
    test_ratio: float = DEFAULT_TEST_SPLIT,
) -> dict[str, list[dict[str, str]]]:
    """Combine all data sources, deduplicate, and split into train/val/test.

    Sources are loaded in order, with deduplication based on the normalized
    (en, tlh) text pair. The OPUS Tatoeba corpus provides the bulk of data,
    supplemented by the Tatoeba API, boQwI' dictionary, paq'batlh epic poem,
    and curated proverbs.

    Returns:
        Dict with keys "train", "val", "test", each a list of {"en", "tlh"} dicts.
    """
    ensure_dirs()

    # Collect from all sources
    print("Collecting data from all sources...\n")
    all_pairs = []

    sources = [
        ("OPUS Tatoeba", load_opus_tatoeba),
        ("Tatoeba API", download_tatoeba),
        ("boQwI' dictionary", parse_boqwi),
        ("paq'batlh", load_paqbatlh),
        ("Proverbs", load_proverbs),
    ]

    source_counts = {}
    for name, loader in sources:
        pairs = loader()
        source_counts[name] = len(pairs)
        all_pairs.extend(pairs)
        print()

    print(f"Raw total: {len(all_pairs)} pairs from {len(sources)} sources")
    for name, count in source_counts.items():
        print(f"  {name}: {count}")

    # Deduplicate by normalized (en, tlh) tuple
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = (pair["en"].lower().strip(), pair["tlh"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    dupes = len(all_pairs) - len(unique_pairs)
    print(f"\nAfter deduplication: {len(unique_pairs)} unique pairs ({dupes} duplicates removed)")

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
    print("=" * 60)
    print("Building Klingon-English dataset")
    print("=" * 60 + "\n")
    splits = build_dataset()
    total = sum(len(v) for v in splits.values())
    print(f"\nDone! Total: {total} pairs")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")
