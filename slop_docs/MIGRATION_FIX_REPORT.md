# Migration Fix Report: Training Scripts Run from Colab

## Summary

- **Stale references:** One path fix (`configs` → `slop_configs`) in `slop_scripts/eval_prompts.py`. No remaining `slop_minimization` or `scripts/` references in `slop_scripts/` or `slop_src/slop/`.
- **Missing package:** Scripts and package code expected `slop.data` (dataset, tokenizer, token_labels). That subpackage did not exist under `slop_src/slop/`. It was added by creating `slop_src/slop/data/` with minimal implementations aligned to the old `slop-minimization` layout.
- **Verification:** All four required entrypoints run from repo root with `PYTHONPATH=<repo>/slop_src` and `python slop_scripts/<script>.py ...`.

---

## 1. Files changed

| File | Change |
|------|--------|
| **slop_scripts/eval_prompts.py** | `root / "configs"` → `root / "slop_configs"` (fallback config path when reward/generator config is missing in run dir). |
| **slop_src/slop/data/__init__.py** | **NEW** – exports `load_jsonl`, `SlopDataset`, `tokenize_and_align_labels`, `SlopTokenizer`, and token_labels helpers. |
| **slop_src/slop/data/dataset.py** | **NEW** – `load_jsonl()`, `SlopDataset` (for token-level classification). |
| **slop_src/slop/data/tokenizer.py** | **NEW** – `tokenize_and_align_labels()`, `SlopTokenizer` (HF tokenizer wrapper). |
| **slop_src/slop/data/token_labels.py** | **NEW** – `detect_sloppy_spans`, `spans_to_token_labels`, `build_token_label_examples`, `DEFAULT_SLOP_PHRASES`. |

No other Python files in `slop_scripts/` or `slop_src/slop/` required edits. Imports already use `slop.*` and default config paths use `slop_configs/`.

---

## 2. Before/after import and path fixes

### eval_prompts.py

- **Before:** `with open(root / "configs" / "prompt_opt.yaml") as f:`
- **After:** `with open(root / "slop_configs" / "prompt_opt.yaml") as f:`

### slop.data (previously missing)

- **Before:** Scripts used `from slop.data.dataset import load_jsonl`, `from slop.data.tokenizer import ...`; package had no `slop/data` → `ModuleNotFoundError: No module named 'slop.data'`.
- **After:** `slop_src/slop/data/` added with:
  - `dataset.py`: `load_jsonl`, `SlopDataset`
  - `tokenizer.py`: `tokenize_and_align_labels`, `SlopTokenizer`
  - `token_labels.py`: span/label helpers for tests and future use
  - `__init__.py`: re-exports the above

---

## 3. Remaining blockers

- **None** for the four Colab entrypoints below. They run from repo root with `PYTHONPATH=/content/slop-repo/slop_src` (or local equivalent).
- **Optional:** `train_token_classifier.py` and `train_slop_generator.py train` require GPU and network (HuggingFace) in Colab. Not a code migration issue.
- **Tests:** `slop_tests/` assume `PYTHONPATH` includes repo root’s `slop_src` (or conftest). With the new `slop.data` package, tests that depend on `slop.data` (e.g. `test_tokenizer.py`, `test_token_labels.py`) should run; no changes were required in test files for this migration.

---

## 4. Colab commands that work

Assume Colab kernel cwd is the repo root (e.g. `/content/slop-repo` after `!git clone ...` and `%cd /content/slop-repo`). Set:

```text
PROJECT_ROOT=/content/slop-repo
```

Then:

```bash
# 1. Build data
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/build_data.py --output-dir data

# 2. Train slop classifier (saves to outputs/classifier_curriculum)
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_token_classifier.py \
  --config slop_configs/classifier_encoder.yaml \
  --output-dir outputs/classifier_curriculum

# 3a. Generate slop pairs for T5
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py generate \
  --input data/train.jsonl --output data/slop_pairs.jsonl

# 3b. Train T5 slop rewriter (saves to outputs/slop_rewriter)
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py train \
  --train-path data/slop_pairs.jsonl \
  --output-dir outputs/slop_rewriter \
  --model-name t5-small --epochs 3

# 4. Prompt optimization (uses slop_configs/prompt_opt.yaml; reward checkpoint = outputs/classifier_curriculum)
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/optimize_prompts.py \
  --config slop_configs/prompt_opt.yaml
```

Single-line form for Colab `!` cells (same order):

```bash
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/build_data.py --output-dir data
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_token_classifier.py --config slop_configs/classifier_encoder.yaml --output-dir outputs/classifier_curriculum
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py generate --input data/train.jsonl --output data/slop_pairs.jsonl
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py train --train-path data/slop_pairs.jsonl --output-dir outputs/slop_rewriter --model-name t5-small --epochs 3
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/optimize_prompts.py --config slop_configs/prompt_opt.yaml
```

Ensure `slop_configs/prompt_opt.yaml` has `reward.checkpoint_path: outputs/classifier_curriculum` (or the path where the classifier was saved) so prompt optimization uses the trained classifier.
