"""Surrogate model backends for trace_xai."""

from .base import BaseSurrogate
from .decision_tree import DecisionTreeSurrogate

__all__ = [
    "BaseSurrogate",
    "DecisionTreeSurrogate",
]
