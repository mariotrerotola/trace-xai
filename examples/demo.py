#!/usr/bin/env python3
"""Demo: explain four different classifiers on the Iris dataset."""

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from trace_xai import Explainer

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
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    model.fit(X, y)

    explainer = Explainer(model, feature_names=feature_names, class_names=class_names)
    result = explainer.extract_rules(X, y=y, max_depth=4)

    print(result.rules)
    print()
    print(result.report)

    save_path = f"tree_{name.lower()}.png"
    result.plot(save_path=save_path)
    print(f"\n  Tree saved to {save_path}")
