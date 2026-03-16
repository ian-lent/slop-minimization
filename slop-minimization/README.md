# Slop Minimization

Production-style repo for token-level slop classification and prompt optimization using PyTorch + HuggingFace Transformers.

## Goals

1. **Token-level slop classifier** – Transformer backbone (LM) + per-token classification head  
2. **LoRA fine-tuning** – PEFT with optional [Unsloth](https://github.com/unslothai/unsloth) for faster LoRA  
3. **Slop generator** – Rewrites good text into sloppier text (hard negatives)  
4. **Prompt optimization** – Black-box prompt search using the classifier as reward model  

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) or [Poetry](https://python-poetry.org/)

## Setup

```bash
cd slop-minimization

# With uv
uv sync

# With poetry
poetry install

# Optional: Unsloth for faster LoRA
uv add "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# or: pip install unsloth

# Optional: wandb for experiment tracking
uv add wandb
# or: poetry add wandb
```

## Project layout

```
slop-minimization/
├── pyproject.toml
├── README.md
├── src/
│   └── slop/
│       ├── config.py          # Config dataclasses
│       ├── scoring.py         # Reward / aggregate scores
│       ├── data/
│       │   ├── tokenizer.py   # Tokenize + align labels
│       │   └── dataset.py     # SlopDataset
│       └── models/
│           ├── token_classifier.py  # LM backbone + per-token head
│           └── slop_generator.py    # Slop text generator
├── scripts/
│   ├── build_data.py
│   ├── train_token_classifier.py
│   ├── train_slop_generator.py
│   ├── optimize_prompts.py
│   └── eval.py
├── configs/
│   ├── default.yaml
│   └── small.yaml
└── tests/
    ├── test_tokenizer.py
    └── test_scoring.py
```

## Commands

All scripts are run from the `slop-minimization/` directory. Set `PYTHONPATH` or run from project root with `python scripts/...`.

### 1. Build data

```bash
# Create data/raw/good.txt and data/raw/slop.txt first, or use placeholders
python scripts/build_data.py \
  --good-path data/raw/good.txt \
  --slop-path data/raw/slop.txt \
  --output-dir data \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

Output: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` with `text` and per-token `labels` (0=good, 1=slop).

### 2. Train token classifier (LoRA)

```bash
python scripts/train_token_classifier.py \
  --train-path data/train.jsonl \
  --val-path data/val.jsonl \
  --output-dir outputs/classifier \
  --model-name gpt2 \
  --batch-size 8 \
  --epochs 3 \
  --max-length 256 \
  --lora-r 16

# With Unsloth (faster LoRA)
python scripts/train_token_classifier.py --use-unsloth --model-name gpt2

# With wandb
python scripts/train_token_classifier.py --use-wandb

# With config file
python scripts/train_token_classifier.py --config configs/small.yaml
```

### 3. Train slop generator

```bash
python scripts/train_slop_generator.py \
  --train-path data/train.jsonl \
  --output-dir outputs/generator \
  --model-name gpt2 \
  --use-unsloth \
  --use-wandb
```

### 4. Optimize prompts (black-box search)

Uses the classifier as reward model: higher reward = less slop.

```bash
python scripts/optimize_prompts.py \
  --classifier-path outputs/classifier \
  --output-path outputs/prompts.txt \
  --num-iterations 100 \
  --population-size 20 \
  --mutation-rate 0.1
```

### 5. Evaluate

```bash
python scripts/eval.py \
  --classifier-path outputs/classifier \
  --test-path data/test.jsonl \
  --output-path outputs/eval_results.json
```

## Config files

- `configs/default.yaml` – Default (Llama-style) config  
- `configs/small.yaml` – GPT-2 for faster iteration  

Override via `--config configs/small.yaml` on scripts that support it.

## Tests

```bash
# From slop-minimization/
uv run pytest tests/ -v
# or: poetry run pytest tests/ -v
```

Tests cover:
- Tokenization and label alignment (`test_tokenizer.py`)
- Scoring and aggregation (`test_scoring.py`)

## WandB (optional)

Set `--use-wandb` on training scripts. Ensure `WANDB_API_KEY` is set if needed.
