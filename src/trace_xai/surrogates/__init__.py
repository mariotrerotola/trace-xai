"""Surrogate model backends for trace_xai."""

from .base import BaseSurrogate
from .decision_tree import DecisionTreeSurrogate
from .oblique_tree import ObliqueTreeSurrogate
from .sparse_oblique_tree import SparseObliqueTreeSurrogate

__all__ = [
    "BaseSurrogate",
    "DecisionTreeSurrogate",
    "ObliqueTreeSurrogate",
    "SparseObliqueTreeSurrogate",
]
