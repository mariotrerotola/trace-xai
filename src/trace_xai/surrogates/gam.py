"""GAM surrogate — placeholder for future implementation."""

from __future__ import annotations


class GAMSurrogate:
    """Placeholder: Generalised Additive Model surrogate.

    Will be implemented in a future version using ``pygam``.
    Install with ``pip install trace_xai[gam]``.
    """

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "GAMSurrogate is not yet implemented. "
            "Install pygam (pip install trace_xai[gam]) "
            "and use 'decision_tree' surrogate for now."
        )
