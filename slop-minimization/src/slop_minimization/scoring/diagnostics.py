"""Anti-reward-hacking diagnostics: expose metrics without blocking scoring."""

from __future__ import annotations

import re
from typing import Any


def repetition_ratio(text: str, n: int = 2) -> float:
    """Fraction of tokens that are part of a repeated n-gram (n tokens repeated)."""
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    seen = set()
    repeat_count = 0
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        if ng in seen:
            repeat_count += n
        else:
            seen.add(ng)
    return repeat_count / max(len(tokens), 1)


def repeated_token_fraction(tokens: list[str]) -> float:
    """Fraction of tokens that are exact duplicates of a previous token."""
    if not tokens:
        return 0.0
    seen = set()
    dup = 0
    for t in tokens:
        if t in seen:
            dup += 1
        else:
            seen.add(t)
    return dup / len(tokens)


def punctuation_ratio(text: str) -> float:
    """Fraction of characters that are punctuation."""
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in ".,;:!?\"'()-[]{}")
    return punct / len(text)


def caps_ratio(text: str) -> float:
    """Fraction of letters that are uppercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def filler_loop_score(text: str, filler_phrases: list[str] | None = None) -> float:
    """Crude score for repeated filler-like phrases (e.g. 'like like like')."""
    if filler_phrases is None:
        filler_phrases = ["like", "you know", "um", "uh", "well", "so", "actually"]
    text_lower = text.lower()
    score = 0.0
    for phrase in filler_phrases:
        # Count repeated adjacent occurrences
        pattern = r"(\b" + re.escape(phrase) + r"\b\s*){2,}"
        for m in re.finditer(pattern, text_lower):
            score += min(1.0, len(m.group(0).split()) / 5.0)
    return min(1.0, score)


def compute_diagnostics(
    text: str,
    token_count: int | None = None,
    filler_phrases: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all diagnostics for a single text. Does not block scoring."""
    tokens = text.split()
    n_tok = token_count if token_count is not None else len(tokens)
    return {
        "length_chars": len(text),
        "length_tokens": n_tok,
        "very_short": n_tok < 3,
        "repetition_ratio": repetition_ratio(text, n=2),
        "repeated_token_fraction": repeated_token_fraction(tokens),
        "punctuation_ratio": punctuation_ratio(text),
        "caps_ratio": caps_ratio(text),
        "filler_loop_score": filler_loop_score(text, filler_phrases),
    }
