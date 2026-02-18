"""RQ2: Rule stability under different tolerance levels.

Parallelises over (dataset, model, tolerance) combinations.
New: adds structural stability metrics alongside Jaccard.
"""

import numpy as np
from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results
from trace_xai import Explainer, compute_structural_stability

# Configuration
DATASETS = ["adult", "german_credit", "compas"]
MODELS = ["rf", "xgb", "mlp"]
DEPTH = 4
BOOTSTRAPS = 200
TOLERANCES = [0, 0.01, 0.1, 0.5]
SEEDS = [42]


def _run_one(dataset_name, model_name, tolerance, seed):
    """Single (dataset, model, tolerance) experiment."""
    X, y, task, feature_names, _ = load_dataset(dataset_name)
    if X is None:
        return {"dataset": dataset_name, "model": model_name,
                "tolerance": tolerance, "seed": seed, "error": "dataset_load_failed"}

    model = get_model(model_name, task, random_state=seed)
    model.fit(X, y)

    explainer = Explainer(model, feature_names=feature_names, task=task)

    # Jaccard stability (original metric)
    stability_report = explainer.compute_stability(
        X,
        n_bootstraps=BOOTSTRAPS,
        max_depth=DEPTH,
        random_state=seed,
        tolerance=tolerance if tolerance > 0 else None,
    )

    # Structural stability (new metric — prediction agreement, feature rank, top-k)
    structural = compute_structural_stability(
        explainer, X, n_bootstraps=min(BOOTSTRAPS, 30), max_depth=DEPTH,
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "tolerance": tolerance,
        "mean_jaccard": stability_report.mean_jaccard,
        "std_jaccard": stability_report.std_jaccard,
        # Structural stability
        "prediction_agreement": structural.mean_prediction_agreement,
        "feature_importance_stability": structural.feature_importance_stability,
        "top_k_feature_agreement": structural.top_k_feature_agreement,
    }


def run_rq2():
    tasks = [
        {"dataset_name": ds, "model_name": m, "tolerance": t, "seed": s}
        for ds in DATASETS
        for m in MODELS
        for t in TOLERANCES
        for s in SEEDS
    ]
    print(f"RQ2: {len(tasks)} experiments to run")
    results = run_parallel(_run_one, tasks, desc="RQ2")
    save_results(results, "rq2_results.json")


if __name__ == "__main__":
    run_rq2()
