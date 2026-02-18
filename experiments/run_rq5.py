"""RQ5: Scalability analysis — time vs N, depth, and ensemble size.

Parallelises experiments A/B/C independently.
"""

import time

import numpy as np
from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results
from trace_xai import Explainer

# Configuration
DATASET = "adult"
MODEL = "rf"
NS = [1000, 5000, 10000, 25000, 48842]
DEPTHS = [2, 4, 6, 8]
ENSEMBLES = [10, 25, 50, 100]
SEED = 42


def _train_model():
    """Train the black-box once (called by each worker — fast for RF)."""
    X, y, task, feature_names, _ = load_dataset(DATASET)
    model = get_model(MODEL, task, random_state=SEED)
    model.fit(X, y)
    return model, X, y, task, feature_names


def _run_vary_n(n):
    """Experiment A: vary dataset size."""
    model, X, y, task, feature_names = _train_model()
    rng = np.random.RandomState(SEED)
    actual_n = min(n, len(X))
    idx = rng.choice(len(X), size=actual_n, replace=False)
    X_sub = X.iloc[idx]

    explainer = Explainer(model, feature_names=feature_names, task=task)
    t0 = time.perf_counter()
    explainer.extract_rules(X_sub, max_depth=4)
    elapsed = time.perf_counter() - t0

    return {
        "vary": "N", "n": actual_n, "depth": 4, "ensemble_size": 1,
        "time_s": elapsed,
    }


def _run_vary_depth(depth):
    """Experiment B: vary surrogate depth."""
    model, X, y, task, feature_names = _train_model()
    explainer = Explainer(model, feature_names=feature_names, task=task)

    t0 = time.perf_counter()
    explainer.extract_rules(X, max_depth=depth)
    elapsed = time.perf_counter() - t0

    return {
        "vary": "Depth", "n": len(X), "depth": depth, "ensemble_size": 1,
        "time_s": elapsed,
    }


def _run_vary_ensemble(ensemble_size):
    """Experiment C: vary ensemble size."""
    model, X, y, task, feature_names = _train_model()
    explainer = Explainer(model, feature_names=feature_names, task=task)

    t0 = time.perf_counter()
    explainer.extract_stable_rules(
        X, n_estimators=ensemble_size, max_depth=4, frequency_threshold=0.5,
    )
    elapsed = time.perf_counter() - t0

    return {
        "vary": "Ensemble", "n": len(X), "depth": 4,
        "ensemble_size": ensemble_size, "time_s": elapsed,
    }


def run_rq5():
    # Run all three experiment types in parallel
    tasks_n = [{"n": n} for n in NS]
    tasks_d = [{"depth": d} for d in DEPTHS]
    tasks_e = [{"ensemble_size": e} for e in ENSEMBLES]

    print(f"RQ5: {len(tasks_n) + len(tasks_d) + len(tasks_e)} experiments")

    results = []
    results.extend(run_parallel(_run_vary_n, tasks_n, desc="RQ5-VaryN"))
    results.extend(run_parallel(_run_vary_depth, tasks_d, desc="RQ5-VaryD"))
    results.extend(run_parallel(_run_vary_ensemble, tasks_e, desc="RQ5-VaryEns"))

    save_results(results, "rq5_results.json")


if __name__ == "__main__":
    run_rq5()
