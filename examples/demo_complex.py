#!/usr/bin/env python3
"""
Advanced demo: test trace_xai on a complex, realistic scenario.

This example generates a multi-class classification problem with
2,000 samples, 15 features (mix of informative, redundant, and noisy), 4
imbalanced classes, and multiple black-box models.

Evaluation covers:
  - Surrogate comparison: decision_tree vs sparse_oblique_tree vs oblique_tree
  - Phantom split analysis per model via SparseObliqueTreeSurrogate
  - Fidelity across different surrogate depths (sparse_oblique_tree)
  - Per-class fidelity analysis
  - Rule filtering and inspection
  - Cross-validated fidelity estimation
  - Complexity vs. fidelity trade-off
"""

import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from trace_xai import Explainer, SparseObliqueTreeSurrogate
from trace_xai.surrogates.oblique_tree import ObliqueTreeSurrogate

warnings.filterwarnings("ignore")

# ── 1. Data generation ──────────────────────────────────────────────────
print("Generating complex synthetic dataset...")

X_raw, y_encoded = make_classification(
    n_samples=2_000,
    n_features=15,
    n_informative=8,
    n_redundant=4,
    n_clusters_per_class=2,
    n_classes=4,
    weights=[0.45, 0.25, 0.20, 0.10],  # imbalanced
    flip_y=0.05,                         # 5% label noise
    random_state=42,
)

scaler = StandardScaler()
X_processed = scaler.fit_transform(X_raw)

feature_names = [f"feat_{i:02d}" for i in range(X_processed.shape[1])]
class_names = ["ClassA", "ClassB", "ClassC", "ClassD"]

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"  Samples: {len(y_encoded)} (train={len(y_train)}, test={len(y_test)})")
print(f"  Features: {X_processed.shape[1]} (8 informative, 4 redundant, 3 noise)")
print(f"  Classes: {class_names}")
print(f"  Class distribution: {dict(zip(class_names, np.bincount(y_encoded)))}")

# ── 2. Define black-box models ──────────────────────────────────────────
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True
    ),
    "SVM-RBF": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
}

# ── 3. Train & explain each model: surrogate comparison ─────────────────
print(f"\n{'=' * 80}")
print(f"  SURROGATE COMPARISON: decision_tree vs sparse_oblique_tree vs oblique_tree")
print(f"{'=' * 80}")

results_summary = []

for name, model in models.items():
    print(f"\n{'─' * 80}")
    print(f"  Model: {name}")
    print(f"{'─' * 80}")

    model.fit(X_train, y_train)
    bb_acc_train = model.score(X_train, y_train)
    bb_acc_test = model.score(X_test, y_test)
    y_bb = model.predict(X_train)
    print(f"  Black-box accuracy: train={bb_acc_train:.4f}, test={bb_acc_test:.4f}")

    explainer = Explainer(model, feature_names=feature_names, class_names=class_names)

    # decision_tree baseline
    result_dt = explainer.extract_rules(X_train, y=y_train, max_depth=5,
                                        surrogate_type="decision_tree")
    fid_dt = result_dt.report.fidelity

    # sparse_oblique_tree
    result_sparse = explainer.extract_rules(X_train, y=y_train, max_depth=5,
                                            surrogate_type="sparse_oblique_tree")
    fid_sparse = result_sparse.report.fidelity
    sparse_surr = result_sparse.surrogate
    n_pairs_sparse = len(sparse_surr.interaction_pairs_)

    # full oblique_tree (reference — all pairs)
    full_surr = ObliqueTreeSurrogate(max_depth=5, min_samples_leaf=5)
    full_surr.fit(X_train, y_bb)
    fid_full = accuracy_score(y_bb, full_surr.predict(X_train))
    n_pairs_full = len(full_surr._interaction_pairs)

    print(f"\n  Surrogate fidelity (train):")
    print(f"    decision_tree:       {fid_dt:.4f}  | interaction pairs: 0")
    print(f"    sparse_oblique_tree: {fid_sparse:.4f}  | interaction pairs: {n_pairs_sparse}  "
          f"(phantom feats: {sparse_surr.phantom_features_}, iter: {sparse_surr.n_iterations_})")
    print(f"    oblique_tree:        {fid_full:.4f}  | interaction pairs: {n_pairs_full} (all)")

    print(f"\n  Sparse surrogate — Fidelity Report:")
    print(result_sparse.report)

    print(f"\n  Sparse surrogate — Per-class fidelity:")
    for cls, fid in result_sparse.report.class_fidelity.items():
        print(f"    {cls}: {fid:.4f}")

    # Top rules per class from sparse surrogate
    for cls in class_names:
        filtered = result_sparse.rules.filter_by_class(cls)
        top_rules = sorted(filtered.rules, key=lambda r: r.samples, reverse=True)[:2]
        print(f"\n  Top 2 rules for '{cls}' (sparse_oblique_tree, by samples):")
        for i, rule in enumerate(top_rules, 1):
            print(f"    {i}. {rule}")

    results_summary.append({
        "model": name,
        "bb_test_acc": bb_acc_test,
        "fid_dt": fid_dt,
        "fid_sparse": fid_sparse,
        "fid_full": fid_full,
        "n_pairs_sparse": n_pairs_sparse,
        "n_pairs_full": n_pairs_full,
        "phantom_feats": len(sparse_surr.phantom_features_),
        "num_rules_sparse": result_sparse.rules.num_rules,
        "avg_len_sparse": result_sparse.report.avg_rule_length,
    })

# ── 4. Depth sweep on sparse_oblique_tree ───────────────────────────────
print(f"\n{'=' * 80}")
print(f"  DEPTH SWEEP — sparse_oblique_tree: fidelity vs. interpretability")
print(f"{'=' * 80}")

sweep_model = models["GradientBoosting"]
depths = [2, 3, 5, 8, None]

sweep_results = []
explainer_gb = Explainer(sweep_model, feature_names=feature_names, class_names=class_names)

for depth in depths:
    depth_label = str(depth) if depth is not None else "unlimited"
    result = explainer_gb.extract_rules(
        X_test, y=y_test,
        max_depth=depth,
        surrogate_type="sparse_oblique_tree",
    )
    sp = result.surrogate
    sweep_results.append({
        "depth": depth_label,
        "fidelity": result.report.fidelity,
        "accuracy": result.report.accuracy,
        "num_rules": result.report.num_rules,
        "avg_length": result.report.avg_rule_length,
        "max_length": result.report.max_rule_length,
        "pairs": len(sp.interaction_pairs_),
        "phantom": len(sp.phantom_features_),
    })

print(f"\n  {'Depth':<12} {'Fidelity':>10} {'Accuracy':>10} {'#Rules':>8} "
      f"{'AvgLen':>8} {'MaxLen':>8} {'Pairs':>7} {'Phantom':>8}")
print(f"  {'─' * 74}")
for r in sweep_results:
    acc_str = f"{r['accuracy']:>10.4f}" if r["accuracy"] is not None else f"{'—':>10}"
    print(f"  {r['depth']:<12} {r['fidelity']:>10.4f} {acc_str} {r['num_rules']:>8} "
          f"{r['avg_length']:>8.1f} {r['max_length']:>8} {r['pairs']:>7} {r['phantom']:>8}")

# ── 5. Cross-validated fidelity (sparse_oblique_tree) ────────────────────
print(f"\n{'=' * 80}")
print(f"  CROSS-VALIDATED FIDELITY — sparse_oblique_tree (GradientBoosting, 3-fold, depth=5)")
print(f"{'=' * 80}")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_fidelities = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_processed, y_encoded), 1):
    X_cv_train, X_cv_val = X_processed[train_idx], X_processed[val_idx]
    y_cv_train, y_cv_val = y_encoded[train_idx], y_encoded[val_idx]

    cv_model = GradientBoostingClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
    )
    cv_model.fit(X_cv_train, y_cv_train)

    cv_explainer = Explainer(cv_model, feature_names=feature_names, class_names=class_names)
    cv_result = cv_explainer.extract_rules(
        X_cv_val, y=y_cv_val,
        max_depth=5,
        surrogate_type="sparse_oblique_tree",
    )
    sp = cv_result.surrogate
    cv_fidelities.append(cv_result.report.fidelity)
    print(f"  Fold {fold}: fidelity={cv_result.report.fidelity:.4f}, "
          f"accuracy={cv_result.report.accuracy:.4f}, "
          f"rules={cv_result.report.num_rules}, "
          f"pairs={len(sp.interaction_pairs_)}, "
          f"phantom_feats={sp.phantom_features_}")

print(f"\n  Mean fidelity: {np.mean(cv_fidelities):.4f} ± {np.std(cv_fidelities):.4f}")

# ── 6. Summary table ────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  SUMMARY: all models — surrogate comparison (depth=5)")
print(f"{'=' * 80}")

print(f"\n  {'Model':<20} {'BB Acc':>8} {'DT Fid':>8} {'Sparse Fid':>11} "
      f"{'Full Fid':>9} {'Pairs(S)':>9} {'Pairs(F)':>9} {'Phantom':>8} {'Rules':>6}")
print(f"  {'─' * 92}")
for r in sorted(results_summary, key=lambda x: x["fid_sparse"], reverse=True):
    print(f"  {r['model']:<20} {r['bb_test_acc']:>8.4f} {r['fid_dt']:>8.4f} "
          f"{r['fid_sparse']:>11.4f} {r['fid_full']:>9.4f} "
          f"{r['n_pairs_sparse']:>9} {r['n_pairs_full']:>9} "
          f"{r['phantom_feats']:>8} {r['num_rules_sparse']:>6}")

print(f"\n  Pairs(S) = interaction pairs in sparse_oblique_tree")
print(f"  Pairs(F) = interaction pairs in oblique_tree (all pairwise)")
print(f"  Phantom  = number of original features flagged as phantom splits")

print(f"\n{'=' * 80}")
print("  Done.")
print(f"{'=' * 80}")
