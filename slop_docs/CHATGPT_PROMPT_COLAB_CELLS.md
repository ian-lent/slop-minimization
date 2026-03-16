# Prompt for ChatGPT: Create Colab Cells (Essential Models + Save Weights)

Copy the sections below into ChatGPT. Replace or remove the placeholder if you attach the actual notebook.

---

## Part 1: Context (paste this first)

I have a Colab notebook that does **setup** well (clone repo, pip install, verify paths). The rest of the notebook has many **experimental** steps that I want to strip down. I need you to produce **only the Colab cells** for the **essential pipeline**, with **training weights saved for the critical models**.

**Repo (current layout):**
- GitHub: `https://github.com/ian-lent/slop-minimization.git`
- After cloning, the project root has: `slop_configs`, `slop_scripts`, `slop_src`, `slop_notebooks`, `slop_docs`, `slop_tests`.
- Scripts live in `slop_scripts/`, Python package in `slop_src/slop`, YAML configs in `slop_configs/`.
- All commands must be run from the **repo root** with `PYTHONPATH` set to the repo root’s `slop_src` (e.g. `PYTHONPATH=/content/slop-repo/slop_src` or `PROJECT_ROOT/slop_src`).

**Setup style to keep (from my existing Colab):**
- Cell 1: Print Torch version, CUDA available, GPU name (and suggest T4 if no GPU).
- Cell 2: Clone repo into a fixed directory (e.g. `/content/slop-repo`), then `%cd` into that directory so the kernel cwd is the repo root. Set a variable `PROJECT_ROOT = "/content/slop-repo"` (or wherever you cloned).
- Cell 3: `!pip -q install ...` for: torch, transformers, datasets, peft, accelerate, pyyaml, tqdm, scikit-learn, sentencepiece.
- Cell 4: Verify: `!pwd`, `!ls slop_configs slop_scripts slop_src` (so we see the distinct top-level folders).

**Essential pipeline only (no experiments):**
1. **Build data**  
   One cell: `python slop_scripts/build_data.py --output-dir data`, then list `data/` and line counts for `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

2. **Train the slop classifier (critical model — weights must be saved)**  
   - Script: `slop_scripts/train_token_classifier.py`.  
   - Config: `slop_configs/classifier_encoder.yaml`.  
   - Use `--output-dir outputs/classifier_curriculum` so the run is reproducible.  
   - This script **already saves** the classifier: `pytorch_model.bin` and tokenizer files under the given `--output-dir`.  
   - One cell: run the command with `PYTHONPATH={PROJECT_ROOT}/slop_src`, then e.g. `!ls -la outputs/classifier_curriculum` to confirm weights and tokenizer are written.

3. **Slop rewriter / slop generator (critical model — weights must be saved)**  
   - Script: `slop_scripts/train_slop_generator.py`.  
   - First: **generate** pairs: subcommand `generate`, `--input data/train.jsonl`, `--output data/slop_pairs.jsonl`.  
   - Then: **train** T5 rewriter: subcommand `train`, `--train-path data/slop_pairs.jsonl`, `--output-dir outputs/slop_rewriter`, e.g. `--model-name t5-small`, `--epochs 3`.  
   - The train step **already saves** the T5 model and tokenizer to `--output-dir` (e.g. `outputs/slop_rewriter`).  
   - Two cells (one for generate, one for train), each using `PYTHONPATH={PROJECT_ROOT}/slop_src`. After train, add a line to confirm: `!ls -la outputs/slop_rewriter`.

4. **Prompt optimization**  
   - Script: `slop_scripts/optimize_prompts.py`, config `slop_configs/prompt_opt.yaml`.  
   - Config should point the reward model at the trained classifier: `checkpoint_path: outputs/classifier_curriculum`.  
   - One cell: run with `PYTHONPATH={PROJECT_ROOT}/slop_src`; no extra weight saving needed for this step (it saves best prompts and config, not a new trained model).

5. **Save critical artifacts (so weights are not lost)**  
   - One cell that zips (or copies) the **two** critical weight directories for download or Drive:  
     - `outputs/classifier_curriculum` (classifier from classifier_factory / train_token_classifier)  
     - `outputs/slop_rewriter` (slop generator / T5 rewriter from train_slop_generator)  
   - Optional: also include `outputs/prompt_opt` (best_prompts.json, etc.) in the zip.  
   - Example: `shutil.make_archive("slop_critical_weights", "zip", "outputs")` then tell user to download the zip, or copy `outputs/classifier_curriculum` and `outputs/slop_rewriter` to Google Drive with clear paths.

**What to exclude (experimental, do not include):**
- Training or comparing **baseline vs curriculum** classifier (keep only one: curriculum with `--output-dir outputs/classifier_curriculum`).
- `compare_reward_checkpoints.py`, `compare_generators.py`, or any multi-generator / multi-checkpoint comparison.
- `build_classifier_dataset.py` with custom target counts (optional; only include if you consider it essential; otherwise keep only `build_data.py`).
- Any inline code that edits YAML in the notebook (prefer a single run with the repo’s `slop_configs/prompt_opt.yaml` pointing at `outputs/classifier_curriculum`).

**Critical point:**  
Training weights **must** be saved for:  
- **Classifier** (train_token_classifier / classifier_factory) → `outputs/classifier_curriculum`  
- **Slop generator** (T5 rewriter from train_slop_generator) → `outputs/slop_rewriter`  

The scripts already write these; the Colab must run them with the correct `--output-dir` and then **persist** those two directories (zip and/or copy to Drive) so the user does not lose the weights when the session ends.

---

## Part 2: What to output

Please output **only the Colab cells** (code and, if needed, short markdown section headers). Use the exact paths and script names above (`slop_scripts/`, `slop_configs/`, `slop_src`, `PROJECT_ROOT`). I will paste these cells into a new Colab. Number or label cells so I can follow the order (Setup 1–4, Build data, Train classifier, Generate slop pairs, Train slop rewriter, Prompt optimization, Save critical weights).

---

## Optional: If you have the Colab file

If you have the Colab notebook (e.g. `AI_slop (1).ipynb`), you can add:

“Here is my current Colab structure [paste or attach the notebook]. Use the **setup** from the first few cells (clone, pip, pwd/ls), but replace everything after that with the minimal essential pipeline and weight-saving steps described above. Output only the replacement cells (and any small tweaks to the setup cells to use `PROJECT_ROOT` and `slop_configs` / `slop_scripts` / `slop_src`).”
