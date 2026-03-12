"""Scoring utilities: use classifier as reward model for prompt optimization."""

from __future__ import annotations

from typing import Any

import torch


def compute_reward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    label_pad_id: int = -100,
) -> torch.Tensor:
    """Compute per-sequence reward: 1 - mean(slop_prob) over non-padding tokens.

    Higher reward = less slop. Use as reward for prompt optimization.
    """
    slop_probs = model.score_tokens(input_ids, attention_mask)
    if attention_mask is not None:
        mask = (attention_mask == 1).float()
        masked_probs = slop_probs * mask
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_slop = (masked_probs.sum(dim=1) / count.squeeze(-1))
    else:
        mean_slop = slop_probs.mean(dim=1)
    reward = 1.0 - mean_slop
    return reward


def aggregate_token_scores(
    slop_probs: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Aggregate per-token slop probabilities into sequence-level score."""
    if attention_mask is not None:
        mask = (attention_mask == 1).float()
        slop_probs = slop_probs * mask
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        if reduction == "mean":
            return (slop_probs.sum(dim=1) / count.squeeze(-1))
        if reduction == "sum":
            return slop_probs.sum(dim=1)
    if reduction == "mean":
        return slop_probs.mean(dim=1)
    return slop_probs.sum(dim=1)
