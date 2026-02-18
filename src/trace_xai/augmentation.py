"""Data augmentation and query synthesis for improved surrogate fidelity.

Addresses the limitation that TRACE uses only predictions on available data
without exploring under-represented regions of the input space (cf. TREPAN, 1996).
"""

from __future__ import annotations

import numpy as np


def perturbation_augmentation(
    X: np.ndarray,
    model,
    *,
    n_neighbors: int = 5,
    noise_scale: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples via local perturbation (LIME-style).

    For each original sample, create *n_neighbors* perturbed copies by adding
    Gaussian noise scaled by feature-wise standard deviation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original input data.
    model : object
        Black-box model with a ``.predict()`` method.
    n_neighbors : int, default 5
        Number of perturbed neighbors per original sample.
    noise_scale : float, default 0.1
        Standard deviation multiplier for the Gaussian noise
        (relative to feature-wise std).
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    X_aug : ndarray
        Original + synthetic samples concatenated.
    y_aug : ndarray
        Black-box predictions for all augmented samples.
    """
    rng = np.random.RandomState(random_state)
    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    synthetic = []
    for _ in range(n_neighbors):
        noise = rng.normal(0, noise_scale, size=X.shape) * feature_std
        synthetic.append(X + noise)

    X_synth = np.vstack(synthetic)
    X_aug = np.vstack([X, X_synth])
    y_aug = np.asarray(model.predict(X_aug))
    return X_aug, y_aug


def boundary_augmentation(
    X: np.ndarray,
    model,
    surrogate,
    *,
    n_samples: int = 200,
    noise_scale: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples near the surrogate's decision boundaries.

    Identifies points where the surrogate's prediction changes between
    neighbouring leaves, then samples around those boundary regions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original input data (used for feature ranges).
    model : object
        Black-box model with a ``.predict()`` method.
    surrogate : fitted sklearn tree
        The current surrogate tree (used to identify boundaries).
    n_samples : int, default 200
        Number of boundary samples to generate.
    noise_scale : float, default 0.05
        Spread of noise around each boundary threshold
        (relative to feature-wise std).
    random_state : int, default 42
        Random seed.

    Returns
    -------
    X_boundary : ndarray
        Synthetic boundary samples.
    y_boundary : ndarray
        Black-box predictions for the boundary samples.
    """
    rng = np.random.RandomState(random_state)
    tree_ = surrogate.tree_
    feature_ids = tree_.feature
    thresholds = tree_.threshold
    children_left = tree_.children_left

    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)

    # Collect internal split points
    split_features = []
    split_thresholds = []
    for node_id in range(tree_.node_count):
        if children_left[node_id] != -1:  # internal node
            split_features.append(feature_ids[node_id])
            split_thresholds.append(thresholds[node_id])

    if not split_features:
        # No splits = single-leaf tree; nothing to sample
        empty = np.empty((0, X.shape[1]))
        return empty, np.empty(0)

    # Generate samples near each split
    boundary_samples = []
    samples_per_split = max(1, n_samples // len(split_features))

    for feat_idx, thresh in zip(split_features, split_thresholds):
        for _ in range(samples_per_split):
            sample = rng.uniform(feature_mins, feature_maxs)
            # Place the split feature near the boundary
            noise = rng.normal(0, noise_scale * feature_std[feat_idx])
            sample[feat_idx] = thresh + noise
            # Clip to feature range
            sample = np.clip(sample, feature_mins, feature_maxs)
            boundary_samples.append(sample)

    X_boundary = np.array(boundary_samples[:n_samples])
    y_boundary = np.asarray(model.predict(X_boundary))
    return X_boundary, y_boundary


def sparse_region_augmentation(
    X: np.ndarray,
    model,
    surrogate,
    *,
    n_samples: int = 200,
    sparsity_quantile: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples in sparse regions of the input space.

    Identifies surrogate leaves with few training samples and generates
    uniform random points within each sparse leaf's hyper-rectangle.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original input data.
    model : object
        Black-box model with a ``.predict()`` method.
    surrogate : fitted sklearn tree
        The current surrogate tree.
    n_samples : int, default 200
        Total number of synthetic samples.
    sparsity_quantile : float, default 0.25
        Leaves with sample counts below this quantile of all leaf counts
        are considered "sparse".
    random_state : int, default 42
        Random seed.

    Returns
    -------
    X_sparse : ndarray
        Synthetic samples from sparse regions.
    y_sparse : ndarray
        Black-box predictions for the sparse samples.
    """
    rng = np.random.RandomState(random_state)
    tree_ = surrogate.tree_
    children_left = tree_.children_left
    children_right = tree_.children_right
    n_node_samples = tree_.n_node_samples
    feature_ids = tree_.feature
    thresholds = tree_.threshold

    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)
    n_features = X.shape[1]

    # Find all leaves and their bounds
    leaves = []
    _find_leaf_bounds(
        0, tree_, feature_ids, thresholds, children_left, children_right,
        feature_mins.copy(), feature_maxs.copy(), n_node_samples, leaves, n_features,
    )

    if not leaves:
        empty = np.empty((0, X.shape[1]))
        return empty, np.empty(0)

    # Identify sparse leaves
    sample_counts = np.array([leaf["samples"] for leaf in leaves])
    threshold_count = np.quantile(sample_counts, sparsity_quantile)
    sparse_leaves = [leaf for leaf in leaves if leaf["samples"] <= threshold_count]

    if not sparse_leaves:
        sparse_leaves = leaves  # fallback: use all leaves

    # Distribute samples across sparse leaves
    samples_per_leaf = max(1, n_samples // len(sparse_leaves))
    synthetic = []

    for leaf in sparse_leaves:
        lmin = leaf["lower"]
        lmax = leaf["upper"]
        # Ensure valid range
        valid = lmax > lmin
        for _ in range(samples_per_leaf):
            sample = np.where(
                valid,
                rng.uniform(lmin, lmax),
                (lmin + lmax) / 2.0,
            )
            synthetic.append(sample)

    X_sparse = np.array(synthetic[:n_samples])
    y_sparse = np.asarray(model.predict(X_sparse))
    return X_sparse, y_sparse


def _find_leaf_bounds(
    node_id, tree_, feature_ids, thresholds, children_left, children_right,
    lower, upper, n_node_samples, leaves, n_features,
):
    """Recursively find leaf nodes and their bounding boxes."""
    if children_left[node_id] == children_right[node_id]:
        # Leaf node
        leaves.append({
            "node_id": node_id,
            "samples": int(n_node_samples[node_id]),
            "lower": lower.copy(),
            "upper": upper.copy(),
        })
        return

    feat = feature_ids[node_id]
    thresh = thresholds[node_id]

    # Left child: feature <= threshold
    left_upper = upper.copy()
    left_upper[feat] = min(left_upper[feat], thresh)
    _find_leaf_bounds(
        children_left[node_id], tree_, feature_ids, thresholds,
        children_left, children_right, lower.copy(), left_upper,
        n_node_samples, leaves, n_features,
    )

    # Right child: feature > threshold
    right_lower = lower.copy()
    right_lower[feat] = max(right_lower[feat], thresh)
    _find_leaf_bounds(
        children_right[node_id], tree_, feature_ids, thresholds,
        children_left, children_right, right_lower, upper.copy(),
        n_node_samples, leaves, n_features,
    )


def augment_data(
    X: np.ndarray,
    model,
    surrogate=None,
    *,
    strategy: str = "perturbation",
    n_neighbors: int = 5,
    noise_scale: float = 0.1,
    n_boundary_samples: int = 200,
    n_sparse_samples: int = 200,
    sparsity_quantile: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Unified augmentation interface.

    Parameters
    ----------
    strategy : str
        One of ``"perturbation"``, ``"boundary"``, ``"sparse"``, or ``"combined"``.
        - ``"perturbation"``: Local Gaussian perturbation (LIME-style).
        - ``"boundary"``: Sample near surrogate decision boundaries.
        - ``"sparse"``: Sample in under-represented leaf regions.
        - ``"combined"``: Apply all three strategies.
    """
    X = np.asarray(X)

    if strategy == "perturbation":
        return perturbation_augmentation(
            X, model, n_neighbors=n_neighbors, noise_scale=noise_scale,
            random_state=random_state,
        )

    if strategy in ("boundary", "sparse", "combined") and surrogate is None:
        raise ValueError(
            f"Strategy '{strategy}' requires a fitted surrogate. "
            "Pass the surrogate parameter or use strategy='perturbation'."
        )

    if strategy == "boundary":
        X_b, y_b = boundary_augmentation(
            X, model, surrogate, n_samples=n_boundary_samples,
            noise_scale=noise_scale, random_state=random_state,
        )
        return np.vstack([X, X_b]), np.concatenate([
            np.asarray(model.predict(X)), y_b,
        ])

    if strategy == "sparse":
        X_s, y_s = sparse_region_augmentation(
            X, model, surrogate, n_samples=n_sparse_samples,
            sparsity_quantile=sparsity_quantile, random_state=random_state,
        )
        return np.vstack([X, X_s]), np.concatenate([
            np.asarray(model.predict(X)), y_s,
        ])

    if strategy == "combined":
        X_p, y_p = perturbation_augmentation(
            X, model, n_neighbors=n_neighbors, noise_scale=noise_scale,
            random_state=random_state,
        )
        X_b, y_b = boundary_augmentation(
            X, model, surrogate, n_samples=n_boundary_samples,
            noise_scale=noise_scale, random_state=random_state + 1,
        )
        X_s, y_s = sparse_region_augmentation(
            X, model, surrogate, n_samples=n_sparse_samples,
            sparsity_quantile=sparsity_quantile, random_state=random_state + 2,
        )
        X_all = np.vstack([X_p, X_b, X_s])
        y_all = np.concatenate([y_p, y_b, y_s])
        return X_all, y_all

    raise ValueError(
        f"Unknown augmentation strategy: {strategy!r}. "
        "Use 'perturbation', 'boundary', 'sparse', or 'combined'."
    )
