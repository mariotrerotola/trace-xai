"""Surrogate-tree visualisation helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for safe headless rendering
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz, plot_tree


def plot_surrogate_tree(
    surrogate,
    feature_names: Sequence[str],
    class_names: Sequence[str] = (),
    *,
    figsize: tuple[int, int] = (20, 10),
    fontsize: int = 9,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Render the surrogate tree using ``sklearn.tree.plot_tree``.

    Parameters
    ----------
    surrogate : DecisionTreeClassifier
        Fitted surrogate tree.
    feature_names, class_names : sequence of str
        Labels for features and target classes.
    figsize : tuple, default (20, 10)
        Matplotlib figure size.
    fontsize : int, default 9
        Font size for node labels.
    save_path : str or None
        If given, save the figure to this path (PNG, PDF, SVG, ...).
    dpi : int, default 150
        Resolution when saving.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_kwargs: dict = dict(
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        fontsize=fontsize,
        ax=ax,
    )
    if class_names:
        plot_kwargs["class_names"] = list(class_names)
    plot_tree(surrogate, **plot_kwargs)
    ax.set_title("Surrogate Decision Tree", fontsize=fontsize + 4)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_dot(
    surrogate,
    feature_names: Sequence[str],
    class_names: Sequence[str] = (),
) -> str:
    """Export the surrogate tree in Graphviz DOT format.

    Returns
    -------
    str
        DOT-language string that can be rendered by ``graphviz`` or ``dot``.
    """
    kwargs: dict = dict(
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
    )
    if class_names:
        kwargs["class_names"] = list(class_names)
    return export_graphviz(surrogate, **kwargs)
