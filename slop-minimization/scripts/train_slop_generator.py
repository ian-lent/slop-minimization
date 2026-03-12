#!/usr/bin/env python3
"""Train slop generator: rewrites good text into sloppier text (hard negatives)."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slop_minimization.data.dataset import load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--train-path", type=str, default="data/train.jsonl")
    p.add_argument("--output-dir", type=str, default="outputs/generator")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--use-unsloth", action="store_true")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, _ = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_length,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
        except ImportError:
            print("Unsloth not installed. Using full fine-tuning with PEFT.")
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
            from peft import get_peft_model, LoraConfig, TaskType
            model = get_peft_model(
                model,
                LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"], task_type=TaskType.CAUSAL_LM),
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        from peft import get_peft_model, LoraConfig, TaskType
        model = get_peft_model(
            model,
            LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"], task_type=TaskType.CAUSAL_LM),
        )

    data = load_jsonl(args.train_path)
    if not data:
        raise FileNotFoundError(f"No data at {args.train_path}. Run build_data.py first.")

    # Use good text as input, slop as target for seq2seq-style training
    texts = [ex["text"] for ex in data if ex.get("labels", [0])[0] == 0][:100]
    if not texts:
        texts = [ex["text"] for ex in data][:100]

    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})

    def lm_tokenize(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors=None,
        )
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        out["labels"] = [
            [tid if tid != pad_id else -100 for tid in ids]
            for ids in out["input_ids"]
        ]
        return out

    tokenized = ds.map(lm_tokenize, batched=True, remove_columns=ds.column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=torch.cuda.is_available(),
            report_to="wandb" if args.use_wandb else "none",
        ),
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Generator saved to {args.output_dir}")


if __name__ == "__main__":
    main()
