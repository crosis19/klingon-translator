"""Colab-specific utilities: Drive mount, SSD copy, JSONL loading."""

import json
import shutil
import time
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of parsed JSON objects.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def copy_data_to_local_ssd(
    project_dir: str | Path,
    local_dir: str | Path = "/content/local_project",
) -> tuple[Path, Path]:
    """Copy processed and raw data from Drive to local SSD for fast I/O.

    Google Drive FUSE is slow (~1-2 MB/s). Copying to /content/ (local SSD)
    eliminates network overhead for all data reads during training.

    Copies:
    - All processed/*.jsonl files (required for training)
    - Raw OPUS, paq'batlh, boQwI' files (optional, for tokenizer training)

    Args:
        project_dir: Path to the project on Google Drive.
        local_dir: Local directory to copy to.

    Returns:
        Tuple of (processed_data_dir, raw_data_dir) on local SSD.
    """
    project_dir = Path(project_dir)
    local_dir = Path(local_dir)

    if local_dir.exists():
        shutil.rmtree(local_dir)

    data_dir = local_dir / "data" / "processed"
    raw_dir = local_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Copying training data to local SSD...")
    t0 = time.time()

    # Copy processed JSONL splits (small, required for training)
    src_processed = project_dir / "data" / "processed"
    for f in src_processed.glob("*.jsonl"):
        shutil.copy2(f, data_dir / f.name)

    # Copy raw files needed for tokenizer training (optional)
    try:
        src_raw = project_dir / "data" / "raw"

        opus_src = src_raw / "opus" / "tatoeba" / "Tatoeba.en-tlh.tlh"
        if opus_src.exists():
            opus_dst = raw_dir / "opus" / "tatoeba"
            opus_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(opus_src, opus_dst / opus_src.name)

        paq_src = src_raw / "paqbatlh_pairs.json"
        if paq_src.exists():
            shutil.copy2(paq_src, raw_dir / paq_src.name)

        mono_src = src_raw / "boqwi_monolingual.txt"
        if mono_src.exists():
            shutil.copy2(mono_src, raw_dir / mono_src.name)
    except (OSError, ConnectionError) as e:
        print(f"  Warning: could not copy raw files from Drive: {e}")
        print("  Tokenizer training will use processed JSONL data only")

    elapsed = time.time() - t0
    print(f"  Copied in {elapsed:.1f}s")

    return data_dir, raw_dir
