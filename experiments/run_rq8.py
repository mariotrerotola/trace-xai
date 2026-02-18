"""RQ8: Phantom split analysis — SparseObliqueTreeSurrogate vs axis-aligned.

Research question: how many axis-aligned splits in a standard decision tree
surrogate are "phantom" (i.e. the black-box model does NOT change its
prediction when crossing the threshold)?  Does the sparse oblique approach
reduce phantom splits while maintaining fidelity?

Metrics collected per (dataset, model, depth):
  - n_internal_nodes: total internal splits in the axis-aligned tree
  - n_phantom_features: distinct features involved in phantom splits
  - phantom_rate: n_phantom_features / n_features
  - fid_dt: decision_tree fidelity (axis-aligned, no interactions)
  - fid_sparse: sparse_oblique_tree fidelity (with phantom-guided interactions)
  - fidelity_gain: fid_sparse - fid_dt
  - n_interaction_pairs: interaction features added by sparse surrogate
  - sparsity_ratio: n_interaction_pairs / n_all_pairs (0 = axis-aligned, 1 = full oblique)
  - n_iterations: phantom detection iterations performed

Parallelises over (dataset, model, depth) combinations.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results
from trace_xai import SparseObliqueTreeSurrogate

# Configuration
DATASETS = ["adult", "german_credit", "compas"]
MODELS = ["rf", "xgb", "mlp"]
DEPTHS = [3, 4, 5, 6]
SEEDS = [42]
N_PROBES = 20
NOISE_SCALE = 0.05
PHANTOM_THRESHOLD = 0.3


def _count_internal_nodes(tree):
    """Count internal (non-leaf) nodes in an sklearn tree."""
    children_left = tree.tree_.children_left
    return int(np.sum(children_left != children_left[0].__class__(-1)))


def _run_one(dataset_name, model_name, depth, seed):
    """Single (dataset, model, depth, seed) phantom analysis experiment."""
    X, y, task, feature_names, _ = load_dataset(dataset_name)
    if X is None:
        return {"dataset": dataset_name, "model": model_name, "depth": depth,
                "seed": seed, "error": "dataset_load_failed"}

    X_arr = np.asarray(X, dtype=float)
    n_features = X_arr.shape[1]
    n_all_pairs = n_features * (n_features - 1) // 2

    model = get_model(model_name, task, random_state=seed)
    model.fit(X, y)
    y_bb = np.asarray(model.predict(X_arr))

    # ── 1. Axis-aligned DT (baseline) ────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5,
                                random_state=seed)
    dt.fit(X_arr, y_bb)
    fid_dt = accuracy_score(y_bb, dt.predict(X_arr))
    n_internal = _count_internal_nodes(dt)

    # ── 2. SparseObliqueTreeSurrogate — phantom detection + sparse oblique
    sparse_surr = SparseObliqueTreeSurrogate(
        max_depth=depth,
        min_samples_leaf=5,
        max_iterations=2,
        phantom_threshold=PHANTOM_THRESHOLD,
        n_probes=N_PROBES,
        noise_scale=NOISE_SCALE,
        random_state=seed,
    )
    sparse_surr.fit(X_arr, y_bb, model=model, feature_names=tuple(feature_names))
    fid_sparse = accuracy_score(y_bb, sparse_surr.predict(X_arr))

    n_phantom = len(sparse_surr.phantom_features_)
    n_pairs = len(sparse_surr.interaction_pairs_)
    phantom_rate = n_phantom / n_features if n_features > 0 else 0.0
    sparsity_ratio = n_pairs / n_all_pairs if n_all_pairs > 0 else 0.0
    fidelity_gain = fid_sparse - fid_dt

    return {
        "dataset": dataset_name,
        "model": model_name,
        "depth": depth,
        "seed": seed,
        "n_features": n_features,
        "n_all_pairs": n_all_pairs,
        # Axis-aligned DT
        "n_internal_nodes": n_internal,
        "fid_dt": float(fid_dt),
        # Phantom analysis
        "n_phantom_features": n_phantom,
        "phantom_features": list(sparse_surr.phantom_features_),
        "phantom_rate": float(phantom_rate),
        # Sparse oblique
        "fid_sparse": float(fid_sparse),
        "fidelity_gain": float(fidelity_gain),
        "n_interaction_pairs": n_pairs,
        "sparsity_ratio": float(sparsity_ratio),
        "n_iterations": sparse_surr.n_iterations_,
        "sparse_depth": sparse_surr.get_depth(),
        "sparse_n_leaves": sparse_surr.get_n_leaves(),
    }


def run_rq8():
    tasks = [
        {"dataset_name": ds, "model_name": m, "depth": d, "seed": s}
        for ds in DATASETS
        for m in MODELS
        for d in DEPTHS
        for s in SEEDS
    ]
    print(f"RQ8 (Phantom Analysis): {len(tasks)} experiments to run")
    results = run_parallel(_run_one, tasks, desc="RQ8")
    save_results(results, "rq8_results.json")

    # Print summary table
    print("\n" + "=" * 110)
    print("  RQ8 Summary: Phantom Split Analysis — DT vs SparseObliqueTree")
    print("=" * 110)
    print(f"  {'Dataset':<15} {'Model':<6} {'D':>3} {'Nodes':>6} "
          f"{'Fid(DT)':>8} {'Fid(S)':>8} {'ΔFid':>7} "
          f"{'Phantom':>8} {'Rate':>6} {'Pairs':>6} {'Sparsity':>9} {'Iter':>5}")
    print("  " + "-" * 98)

    for r in sorted(results,
                    key=lambda x: (x.get("dataset", ""), x.get("model", ""), x.get("depth", 0))):
        if not isinstance(r, dict) or "error" in r:
            continue
        delta = r["fidelity_gain"]
        delta_str = f"{delta:>+7.4f}"
        print(f"  {r['dataset']:<15} {r['model']:<6} {r['depth']:>3} "
              f"{r['n_internal_nodes']:>6} "
              f"{r['fid_dt']:>8.4f} {r['fid_sparse']:>8.4f} {delta_str} "
              f"{r['n_phantom_features']:>8} {r['phantom_rate']:>6.2f} "
              f"{r['n_interaction_pairs']:>6} {r['sparsity_ratio']:>9.3f} "
              f"{r['n_iterations']:>5}")

    print("=" * 110)

    # Per-model aggregation
    print("\n  Aggregated by model (mean over datasets and depths):")
    print(f"  {'Model':<8} {'ΔFid μ':>8} {'Phantom Rate μ':>15} {'Sparsity μ':>11}")
    print("  " + "-" * 44)
    for m in MODELS:
        rows = [r for r in results
                if isinstance(r, dict) and "error" not in r and r.get("model") == m]
        if not rows:
            continue
        delta_m = float(np.mean([r["fidelity_gain"] for r in rows]))
        phant_m = float(np.mean([r["phantom_rate"] for r in rows]))
        sparse_m = float(np.mean([r["sparsity_ratio"] for r in rows]))
        print(f"  {m:<8} {delta_m:>+8.4f} {phant_m:>15.3f} {sparse_m:>11.3f}")

    print("=" * 110)


if __name__ == "__main__":
    run_rq8()
