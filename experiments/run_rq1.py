"""RQ1: Impact of tree depth on fidelity, complexity, and theoretical bounds.

Compares three surrogate types at each depth:
  - decision_tree  (standard axis-aligned)
  - sparse_oblique_tree  (phantom-guided interaction selection)

Includes theoretical fidelity bounds and complementary metrics.
Parallelises over (dataset, model, depth) combinations.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results
from trace_xai import Explainer, SparseObliqueTreeSurrogate, compute_fidelity_bounds

# Configuration
DATASETS = ["adult", "german_credit", "compas"]
MODELS = ["rf", "xgb", "mlp"]
DEPTHS = [2, 3, 4, 5, 6, 8]
SEEDS = [42]


def _run_one(dataset_name, model_name, depth, seed):
    """Single (dataset, model, depth, seed) experiment — runs in a worker thread."""
    X, y, task, feature_names, _ = load_dataset(dataset_name)
    if X is None:
        return {"dataset": dataset_name, "model": model_name, "depth": depth,
                "seed": seed, "error": "dataset_load_failed"}

    X_arr = np.asarray(X, dtype=float)

    model = get_model(model_name, task, random_state=seed)
    model.fit(X, y)
    y_bb = np.asarray(model.predict(X_arr))

    explainer = Explainer(model, feature_names=feature_names, task=task)

    # ── decision_tree: 5-fold CV fidelity ──────────────────────────────
    cv_report = explainer.cross_validate_fidelity(
        X_arr, y=y_bb, n_folds=5, max_depth=depth, random_state=seed,
    )

    avg_R_dt = float(np.mean([r.num_rules for r in cv_report.fold_reports]))
    avg_L_dt = float(np.mean([r.avg_rule_length for r in cv_report.fold_reports]))

    # Theoretical bounds (decision_tree)
    avg_depth_dt = int(np.mean([r.surrogate_depth for r in cv_report.fold_reports]))
    avg_samples = int(np.mean([r.num_samples for r in cv_report.fold_reports]))
    bounds = compute_fidelity_bounds(
        depth=avg_depth_dt,
        n_features=len(feature_names),
        n_samples=avg_samples,
        empirical_infidelity=1.0 - cv_report.mean_fidelity,
    )

    # ── sparse_oblique_tree: train-set fidelity ─────────────────────────
    # (CV for sparse would be very slow with probing; use train-set comparison)
    sparse_surr = SparseObliqueTreeSurrogate(
        max_depth=depth,
        min_samples_leaf=5,
        max_iterations=2,
        phantom_threshold=0.3,
        n_probes=15,
        noise_scale=0.05,
        random_state=seed,
    )
    sparse_surr.fit(X_arr, y_bb, model=model, feature_names=tuple(feature_names))
    fid_sparse = accuracy_score(y_bb, sparse_surr.predict(X_arr))

    return {
        "dataset": dataset_name,
        "model": model_name,
        "depth": depth,
        "seed": seed,
        # decision_tree (CV)
        "dt_fidelity_mean": cv_report.mean_fidelity,
        "dt_fidelity_std": cv_report.std_fidelity,
        "dt_num_rules": avg_R_dt,
        "dt_avg_rule_length": avg_L_dt,
        # Theoretical bounds
        "vc_dimension": bounds.vc_dimension,
        "estimation_error_pac": bounds.estimation_error_pac,
        "estimation_error_rademacher": bounds.estimation_error_rademacher,
        "min_fidelity_pac": bounds.min_fidelity_pac,
        "min_fidelity_rademacher": bounds.min_fidelity_rademacher,
        "sample_complexity_eps005": bounds.sample_complexity_required,
        # sparse_oblique_tree
        "sparse_fidelity": fid_sparse,
        "sparse_n_phantom": len(sparse_surr.phantom_features_),
        "sparse_n_pairs": len(sparse_surr.interaction_pairs_),
        "sparse_n_iterations": sparse_surr.n_iterations_,
        "sparse_depth": sparse_surr.get_depth(),
        "sparse_n_leaves": sparse_surr.get_n_leaves(),
    }


def run_rq1():
    tasks = [
        {"dataset_name": ds, "model_name": m, "depth": d, "seed": s}
        for ds in DATASETS
        for m in MODELS
        for d in DEPTHS
        for s in SEEDS
    ]
    print(f"RQ1: {len(tasks)} experiments to run")
    results = run_parallel(_run_one, tasks, desc="RQ1")
    save_results(results, "rq1_results.json")

    # Print summary table
    print("\n" + "=" * 100)
    print("  RQ1 Summary: Fidelity vs Depth — DT vs Sparse-Oblique")
    print("=" * 100)
    print(f"  {'Dataset':<15} {'Model':<6} {'Depth':>6} "
          f"{'DT Fid μ':>9} {'DT Fid σ':>9} "
          f"{'Sparse Fid':>11} {'Phantom':>8} {'Pairs':>6} {'PAC LB':>8}")
    print("  " + "-" * 88)
    for r in sorted(results,
                    key=lambda x: (x.get("dataset", ""), x.get("model", ""), x.get("depth", 0))):
        if not isinstance(r, dict) or "error" in r:
            continue
        print(f"  {r['dataset']:<15} {r['model']:<6} {r['depth']:>6} "
              f"{r['dt_fidelity_mean']:>9.4f} {r['dt_fidelity_std']:>9.4f} "
              f"{r['sparse_fidelity']:>11.4f} "
              f"{r['sparse_n_phantom']:>8} {r['sparse_n_pairs']:>6} "
              f"{r['min_fidelity_pac']:>8.4f}")
    print("=" * 100)


if __name__ == "__main__":
    run_rq1()
