"""Project paths, model IDs, and constants."""

from pathlib import Path

# Project root (two levels up from this file: utils/ -> klingon_translator/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Base model from Hugging Face
BASE_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Language codes (NLLB format)
ENGLISH_CODE = "eng_Latn"
KLINGON_CODE = "tlh_Latn"  # Custom code we'll register

# Data source URLs
BOQWI_REPO_URL = "https://github.com/De7vID/klingon-assistant-data"

# Training defaults
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1


def ensure_dirs() -> None:
    """Create data and model directories if they don't exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
