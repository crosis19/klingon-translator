# Klingon Translator

English <-> Klingon machine translation using a fine-tuned NLLB-200 model.

## Project Structure

- `src/klingon_translator/` - Main Python package
  - `model/translator.py` - Core `KlingonTranslator` class (the main interface)
  - `model/tokenizer.py` - Klingon tokenizer extension for NLLB
  - `data/download.py` - Data acquisition and preprocessing pipeline
  - `utils/config.py` - Paths, model IDs, and constants
- `notebooks/` - Local Jupyter notebooks for experimentation
- `colab/` - Google Colab notebooks for GPU training
- `tests/` - Pytest test files
- `data/` - Downloaded data (gitignored)
- `models/` - Saved model checkpoints (gitignored)

## Environment

- Python 3.12 via Conda (`conda activate klingon`)
- PyTorch CPU locally, GPU on Google Colab
- Base model: `facebook/nllb-200-distilled-600M`

## Conventions

- Format with ruff (line length 88)
- Type hints on public functions
- Tests in `tests/` using pytest
- Notebooks are for experimentation; reusable logic belongs in `src/`

## Key Commands

```bash
conda activate klingon
pip install -e .                    # Install package in dev mode
pytest tests/ -v                    # Run tests
python -m klingon_translator.data.download  # Download training data
```
