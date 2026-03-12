#!/usr/bin/env python3
"""Black-box prompt search: use classifier as reward model to optimize prompts."""

import argparse
import random
import string
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--classifier-path", type=str, default="outputs/classifier")
    p.add_argument("--output-path", type=str, default="outputs/prompts.txt")
    p.add_argument("--num-iterations", type=int, default=100)
    p.add_argument("--population-size", type=int, default=20)
    p.add_argument("--mutation-rate", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--prompt-length", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def random_prompt(length: int) -> str:
    return "".join(random.choices(string.ascii_letters + " ", k=length))


def mutate(prompt: str, rate: float) -> str:
    chars = list(prompt)
    for i in range(len(chars)):
        if random.random() < rate:
            chars[i] = random.choice(string.ascii_letters + " ")
    return "".join(chars)


def score_prompts(model, tokenizer, prompts: list[str], device: str) -> list[float]:
    """Score prompts using classifier as reward (1 - mean slop prob)."""
    if not hasattr(model, "score_tokens"):
        # Fallback: random scores for skeleton
        return [random.random() for _ in prompts]
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    probs = model.score_tokens(inputs["input_ids"], inputs["attention_mask"])
    rewards = 1.0 - probs.mean(dim=1).cpu().tolist()
    return rewards


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    classifier_path = Path(args.classifier_path)
    if not classifier_path.exists():
        print(f"Classifier not found at {classifier_path}. Run train_token_classifier.py first.")
        print("Running black-box search with random rewards (skeleton mode).")

    model = None
    tokenizer = None
    if classifier_path.exists():
        try:
            from slop_minimization.models.token_classifier import SlopTokenClassifier
            tokenizer = AutoTokenizer.from_pretrained(classifier_path)
            model = SlopTokenClassifier(
                backbone_name=str(classifier_path),
                num_labels=2,
            ).to(device)
            model.load_state_dict(torch.load(classifier_path / "pytorch_model.bin", map_location=device))
        except Exception as e:
            print(f"Could not load classifier: {e}. Using random rewards.")

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    population = [random_prompt(args.prompt_length) for _ in range(args.population_size)]
    best_reward = -float("inf")
    best_prompt = population[0]

    for it in range(args.num_iterations):
        scores = score_prompts(model, tokenizer, population, device) if model else [random.random() for _ in population]
        indexed = list(zip(scores, population))
        indexed.sort(key=lambda x: x[0], reverse=True)
        top = [p for _, p in indexed[: args.top_k]]
        best_reward = max(best_reward, indexed[0][0])
        best_prompt = indexed[0][1]

        # New population: top-k + mutations
        new_pop = list(top)
        while len(new_pop) < args.population_size:
            parent = random.choice(top)
            new_pop.append(mutate(parent, args.mutation_rate))
        population = new_pop

        if (it + 1) % 10 == 0:
            print(f"Iter {it+1}/{args.num_iterations} best_reward={best_reward:.4f} best_prompt={best_prompt[:50]}...")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write(best_prompt + "\n")
        for p in population[: args.top_k]:
            f.write(p + "\n")
    print(f"Top prompts saved to {args.output_path}")


if __name__ == "__main__":
    main()
