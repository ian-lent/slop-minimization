"""Models for slop classification and generation."""

from .token_classifier import SlopTokenClassifier
from .slop_generator import SlopGenerator

__all__ = ["SlopTokenClassifier", "SlopGenerator"]
