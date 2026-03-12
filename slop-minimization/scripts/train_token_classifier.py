#!/usr/bin/env python3
"""Train token-level slop classifier with LoRA (optionally Unsloth)."""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Add src to path when run as script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slop_minimization.config import Config
from slop_minimization.data import SlopDataset, SlopTokenizer
from slop_minimization.data.dataset import load_jsonl
from slop_minimization.models.token_classifier import SlopTokenClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to config YAML")
    p.add_argument("--train-path", type=str, default="data/train.jsonl")
    p.add_argument("--val-path", type=str, default="data/val.jsonl")
    p.add_argument("--output-dir", type=str, default="outputs/classifier")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for faster LoRA")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    return p.parse_args()


def get_model_and_tokenizer(args: argparse.Namespace, config: Config | None):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Note: Unsloth targets causal LM generation; classifier uses PEFT LoRA on SlopTokenClassifier.
    if args.use_unsloth:
        print("Unsloth is available for train_slop_generator; classifier uses PEFT LoRA.")

    model = SlopTokenClassifier(
        backbone_name=args.model_name,
        num_labels=2,
        max_length=args.max_length,
    )
    # Apply PEFT LoRA
    from peft import get_peft_model, LoraConfig, TaskType
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn"] if "gpt2" in args.model_name else ["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config) if args.config else None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer = get_model_and_tokenizer(args, config)

    train_data = load_jsonl(args.train_path)
    val_data = load_jsonl(args.val_path)
    if not train_data:
        raise FileNotFoundError(f"No training data at {args.train_path}. Run build_data.py first.")

    from datasets import Dataset
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data) if val_data else None

    from slop_minimization.data.tokenizer import tokenize_and_align_labels

    def tokenize_fn(examples):
        return tokenize_and_align_labels(
            examples,
            tokenizer,
            text_column="text",
            label_column="labels",
            max_length=args.max_length,
        )

    tokenized_train = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names) if val_ds else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps" if tokenized_val else "no",
        eval_steps=100,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
