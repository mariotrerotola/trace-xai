#!/usr/bin/env python3
"""
Advanced demo: test trace_xai on a complex, realistic scenario.

This example generates a multi-class classification problem with
2,000 samples, 15 features (mix of informative, redundant, and noisy), 4
imbalanced classes, and multiple black-box models.

Evaluation covers:
  - Fidelity across different surrogate depths
  - Per-class fidelity analysis (important with imbalanced classes)
  - Rule filtering and inspection
  - Cross-validated fidelity estimation
  - Comparison of rule complexity vs. fidelity trade-off
"""

import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from trace_xai import Explainer

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

# ── 3. Train & explain each model ───────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  TRAINING & EXPLANATION (surrogate depth=5)")
print(f"{'=' * 80}")

results_summary = []

for name, model in models.items():
    print(f"\n{'─' * 80}")
    print(f"  Model: {name}")
    print(f"{'─' * 80}")

    model.fit(X_train, y_train)
    bb_acc_train = model.score(X_train, y_train)
    bb_acc_test = model.score(X_test, y_test)
    print(f"  Black-box accuracy: train={bb_acc_train:.4f}, test={bb_acc_test:.4f}")

    explainer = Explainer(model, feature_names=feature_names, class_names=class_names)
    result = explainer.extract_rules(X_train, y=y_train, max_depth=5)

    print(f"\n  --- Fidelity Report ---")
    print(result.report)

    print(f"\n  --- Per-class fidelity ---")
    for cls, fid in result.report.class_fidelity.items():
        print(f"    {cls}: {fid:.4f}")

    # Show top rules for each class
    for cls in class_names:
        filtered = result.rules.filter_by_class(cls)
        top_rules = sorted(filtered.rules, key=lambda r: r.samples, reverse=True)[:3]
        print(f"\n  --- Top 3 rules for class '{cls}' (by samples) ---")
        for i, rule in enumerate(top_rules, 1):
            print(f"    {i}. {rule}")

    results_summary.append({
        "model": name,
        "bb_test_acc": bb_acc_test,
        "fidelity": result.report.fidelity,
        "surrogate_acc": result.report.accuracy,
        "num_rules": result.report.num_rules,
        "avg_length": result.report.avg_rule_length,
    })

# ── 4. Depth sweep: fidelity vs. interpretability trade-off ─────────────
print(f"\n{'=' * 80}")
print(f"  DEPTH SWEEP: fidelity vs. interpretability")
print(f"{'=' * 80}")

# Use GradientBoosting as representative model
sweep_model = models["GradientBoosting"]
depths = [2, 3, 5, 8, None]  # None = unlimited depth

sweep_results = []
explainer = Explainer(sweep_model, feature_names=feature_names, class_names=class_names)

for depth in depths:
    depth_label = str(depth) if depth is not None else "unlimited"
    result = explainer.extract_rules(X_test, y=y_test, max_depth=depth)
    sweep_results.append({
        "depth": depth_label,
        "fidelity": result.report.fidelity,
        "accuracy": result.report.accuracy,
        "num_rules": result.report.num_rules,
        "avg_length": result.report.avg_rule_length,
        "max_length": result.report.max_rule_length,
    })

print(f"\n  {'Depth':<12} {'Fidelity':>10} {'Accuracy':>10} {'#Rules':>8} {'AvgLen':>8} {'MaxLen':>8}")
print(f"  {'─' * 58}")
for r in sweep_results:
    print(f"  {r['depth']:<12} {r['fidelity']:>10.4f} {r['accuracy']:>10.4f} {r['num_rules']:>8} {r['avg_length']:>8.1f} {r['max_length']:>8}")

# ── 5. Cross-validated fidelity estimation ──────────────────────────────
print(f"\n{'=' * 80}")
print(f"  CROSS-VALIDATED FIDELITY (GradientBoosting, 3-fold, depth=5)")
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
    cv_result = cv_explainer.extract_rules(X_cv_val, y=y_cv_val, max_depth=5)
    cv_fidelities.append(cv_result.report.fidelity)
    print(f"  Fold {fold}: fidelity={cv_result.report.fidelity:.4f}, "
          f"accuracy={cv_result.report.accuracy:.4f}, "
          f"rules={cv_result.report.num_rules}")

print(f"\n  Mean fidelity: {np.mean(cv_fidelities):.4f} ± {np.std(cv_fidelities):.4f}")

# ── 6. Summary table ────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  SUMMARY: all models (surrogate depth=5)")
print(f"{'=' * 80}")

print(f"\n  {'Model':<20} {'BB Test Acc':>12} {'Fidelity':>10} {'Surr Acc':>10} {'#Rules':>8} {'AvgLen':>8}")
print(f"  {'─' * 70}")
for r in sorted(results_summary, key=lambda x: x["fidelity"], reverse=True):
    print(f"  {r['model']:<20} {r['bb_test_acc']:>12.4f} {r['fidelity']:>10.4f} "
          f"{r['surrogate_acc']:>10.4f} {r['num_rules']:>8} {r['avg_length']:>8.1f}")

print(f"\n{'=' * 80}")
print("  Done.")
print(f"{'=' * 80}")
