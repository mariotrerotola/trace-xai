#!/usr/bin/env python3
"""Demo: explain four different classifiers on the Iris dataset.

Shows three surrogate types side by side for each model:
  - decision_tree  (standard axis-aligned)
  - sparse_oblique_tree  (phantom-guided interaction selection — new)
  - oblique_tree   (all pairwise interactions)

The sparse variant is highlighted: it detects phantom splits via
counterfactual probing and adds interactions only where needed,
yielding comparable fidelity with fewer (or equally few) interaction terms.
"""

import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from trace_xai import Explainer, SparseObliqueTreeSurrogate
from trace_xai.surrogates.oblique_tree import ObliqueTreeSurrogate

# ── Data ────────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = list(iris.feature_names)
class_names = list(iris.target_names)

# ── Models to explain ───────────────────────────────────────────────────
models = {
    "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0),
    "SVM": SVC(kernel="rbf", random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=0),
}

# Try adding XGBoost if installed
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, use_label_encoder=False, eval_metric="mlogloss", random_state=0
    )
except ImportError:
    pass

# ── Explain each model ─────────────────────────────────────────────────
for name, model in models.items():
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")

    model.fit(X, y)
    y_bb = model.predict(X)

    explainer = Explainer(model, feature_names=feature_names, class_names=class_names)

    # ── 1. Standard decision_tree surrogate ──────────────────────────────
    result_dt = explainer.extract_rules(X, y=y, max_depth=4,
                                        surrogate_type="decision_tree")
    print(f"\n[decision_tree]  fidelity={result_dt.report.fidelity:.4f}  "
          f"rules={result_dt.rules.num_rules}  "
          f"depth={result_dt.report.surrogate_depth}")

    # ── 2. SparseObliqueTreeSurrogate ────────────────────────────────────
    result_sparse = explainer.extract_rules(X, y=y, max_depth=4,
                                            surrogate_type="sparse_oblique_tree")
    sparse_surr = result_sparse.surrogate
    n_pairs_sparse = len(sparse_surr.interaction_pairs_)
    print(f"[sparse_oblique] fidelity={result_sparse.report.fidelity:.4f}  "
          f"rules={result_sparse.rules.num_rules}  "
          f"depth={result_sparse.report.surrogate_depth}  "
          f"pairs={n_pairs_sparse}  "
          f"phantom_feats={sparse_surr.phantom_features_}  "
          f"iterations={sparse_surr.n_iterations_}")

    # ── 3. Full ObliqueTreeSurrogate (all pairs) for reference ───────────
    full_surr = ObliqueTreeSurrogate(max_depth=4, min_samples_leaf=5)
    full_surr.fit(X, y_bb)
    fid_full = accuracy_score(y_bb, full_surr.predict(X))
    n_pairs_full = len(full_surr._interaction_pairs)
    print(f"[oblique_tree]   fidelity={fid_full:.4f}  "
          f"pairs={n_pairs_full}  (all pairwise — reference only)")

    # ── 4. Print sparse rules ─────────────────────────────────────────────
    print(f"\n  Rules (sparse_oblique_tree):")
    print(result_sparse.rules)
    print()
    print(result_sparse.report)

    # Plot the standard DT tree (sparse_oblique uses an internal sklearn tree
    # with augmented feature names; use result_dt for the plain tree plot)
    save_path = f"tree_{name.lower()}.png"
    result_dt.plot(save_path=save_path)
    print(f"\n  Tree saved to {save_path}")
