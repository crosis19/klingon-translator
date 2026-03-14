# Klingon Translator

English ↔ Klingon machine translation using a fine-tuned [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) model.

## Overview

This project fine-tunes Facebook's NLLB-200 (distilled 600M) multilingual translation model to support Klingon (`tlh_Latn`). The approach includes:

- **SentencePiece tokenizer extension** with byte fallback, split digits, and apostrophe preservation for Klingon morphology
- **Bidirectional training** (English→Klingon and Klingon→English)
- **Transfer learning** via base-tokenizer decomposition for embedding initialization
- **~22,600 parallel sentence pairs** from Tatoeba, OPUS, boQwI' dictionary, paq'batlh, and curated proverbs

## Quick Start

### Prerequisites

- Python 3.12 (via Conda recommended)
- PyTorch

### Setup

```bash
conda create -n klingon python=3.12
conda activate klingon
pip install -e .
```

### Download Training Data

```bash
python -m klingon_translator.data.download
```

### Train (Google Colab)

Open `colab/klingon_training.ipynb` in Google Colab with a GPU runtime. The notebook handles:
1. Installing the package from Google Drive
2. Extending the tokenizer with Klingon
3. Building the dataset
4. Training with configurable hyperparameters
5. Evaluation (BLEU, chrF, sample translations)
6. Saving the fine-tuned model

### Run the Web UI

```bash
python app.py
```

Opens a Gradio interface at `http://localhost:7860` for interactive translation.

### Use in Code

```python
from klingon_translator.model.translator import KlingonTranslator

translator = KlingonTranslator()
print(translator.to_klingon("Today is a good day to die."))
# Heghlu'meH QaQ jajvam.

print(translator.to_english("nuqneH?"))
# What do you want?
```

## Project Structure

```
src/klingon_translator/
  model/
    translator.py     # KlingonTranslator class (main interface)
    tokenizer.py      # SentencePiece tokenizer extension for NLLB
  training/
    gpu.py            # GPU info and hardware config
    trainer.py        # TrainingConfig, trainer builder, model saving
    dataset.py        # BilingualDataset with pre-tokenization
    evaluate.py       # BLEU/chrF metrics, sample translations, reports
    colab_utils.py    # Colab helpers (data copy, JSONL loading)
  data/
    download.py       # Data acquisition and preprocessing pipeline
  utils/
    config.py         # Paths, model IDs, and constants
app.py                # Gradio web UI
colab/
  klingon_training.ipynb  # Google Colab training notebook
tests/                # Pytest test suite
```

## Data Sources

- [Tatoeba/OPUS](https://opus.nlpl.eu/) — parallel sentence pairs
- [boQwI' dictionary](https://github.com/De7vID/klingon-assistant-data) — dictionary entries and sentence examples
- Curated Klingon proverbs and canonical phrases

## Development

```bash
conda activate klingon
pytest tests/ -v          # Run tests
ruff check src/ tests/    # Lint
```

## License

MIT
