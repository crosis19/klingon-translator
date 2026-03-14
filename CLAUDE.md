# Klingon Translator

English <-> Klingon machine translation using a fine-tuned NLLB-200 model.

## Project Structure

- `src/klingon_translator/` - Main Python package
  - `model/translator.py` - Core `KlingonTranslator` class (the main interface)
  - `model/tokenizer.py` - Klingon tokenizer extension for NLLB
  - `data/download.py` - Data acquisition and preprocessing pipeline
  - `training/` - Training subpackage (used by Colab notebook)
    - `gpu.py` - GPU info and hardware config dataclass
    - `trainer.py` - TrainingConfig, Seq2SeqTrainer builder, model saving
    - `dataset.py` - BilingualDataset with pre-tokenization
    - `evaluate.py` - BLEU/chrF metrics, sample translations, reports
    - `colab_utils.py` - Colab helpers (data copy, JSONL loading)
  - `utils/config.py` - Paths, model IDs, and constants
- `app.py` - Gradio web UI for interactive translation
- `colab/klingon_training.ipynb` - Google Colab training notebook
- `tests/` - Pytest test files
- `data/` - Downloaded data (gitignored)
- `models/` - Saved model checkpoints (gitignored)
- `klingon-translator.code-workspace` - VS Code workspace config

## Environment

- Python 3.12 via Conda (`conda activate klingon`)
- PyTorch CPU locally, GPU on Google Colab
- Base model: `facebook/nllb-200-distilled-600M`

## Conventions

- Format with ruff (line length 88)
- Type hints on public functions
- Tests in `tests/` using pytest
- Reusable logic belongs in `src/`; the Colab notebook is a thin orchestration layer

## Key Commands

```bash
conda activate klingon
pip install -e .                    # Install package in dev mode
pytest tests/ -v                    # Run tests
python -m klingon_translator.data.download  # Download training data
python app.py                       # Launch Gradio web UI
```
