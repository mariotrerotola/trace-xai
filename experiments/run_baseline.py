"""Baseline comparison: TRACE (DT) vs Sparse-Oblique vs Single DT vs Anchors vs CORELS.

Parallelises over (dataset, model) combinations.
Includes:
  - TRACE with standard decision_tree surrogate
  - TRACE with sparse_oblique_tree surrogate (phantom-guided interactions)
  - Standalone DT (sklearn)
  - Anchors (optional)
  - CORELS (optional)
  - Theoretical fidelity bounds
  - Phantom split analysis per dataset/model
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results
from trace_xai import Explainer, SparseObliqueTreeSurrogate, compute_fidelity_bounds

# Configuration
DATASETS = ["adult", "german_credit", "compas"]
MODELS = ["rf"]
SEEDS = [42]
DEPTH = 4


def _run_anchors(X_arr, model, feature_names):
    """Attempt Anchors — returns (fidelity, n_rules, avg_length) or Nones."""
    try:
        from anchor import anchor_tabular
        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=["0", "1"],
            feature_names=feature_names,
            train_data=X_arr,
            categorical_names={},
        )
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(X_arr), size=min(50, len(X_arr)), replace=False)
        precisions, lengths = [], []
        for i in idxs:
            exp = explainer.explain_instance(X_arr[i], model.predict, threshold=0.95)
            precisions.append(exp.precision())
            lengths.append(len(exp.features()))
        return float(np.mean(precisions)), len(idxs), float(np.mean(lengths))
    except Exception:
        return None, None, None


def _run_corels(X_arr, y_bb, feature_names):
    """Attempt CORELS — returns (fidelity, n_rules, avg_length) or Nones."""
    try:
        from corels import CorelsClassifier
        X_bin = (X_arr > np.median(X_arr, axis=0)).astype(int)
        c = CorelsClassifier(max_card=2, min_support=0.01, verbosity=[])
        bin_names = [f"{f}_high" for f in feature_names]
        c.fit(X_bin, y_bb, features=bin_names)
        preds = c.predict(X_bin)
        fid = accuracy_score(y_bb, preds)
        rl = c.rl()
        return fid, len(rl), float(np.mean([len(r) for r in rl])) if rl else 0.0
    except Exception:
        return None, None, None


def _run_one(dataset_name, model_name, seed):
    """Single (dataset, model) baseline comparison."""
    X, y, task, feature_names, _ = load_dataset(dataset_name)
    if X is None:
        return {"dataset": dataset_name, "model": model_name, "seed": seed,
                "error": "dataset_load_failed"}

    X_arr = np.asarray(X, dtype=float)

    model = get_model(model_name, task, random_state=seed)
    model.fit(X, y)
    y_bb = np.asarray(model.predict(X_arr))

    explainer = Explainer(model, feature_names=feature_names, task=task)

    # 1. TRACE — standard decision_tree surrogate
    res_trace = explainer.extract_rules(X_arr, max_depth=DEPTH,
                                        surrogate_type="decision_tree")
    bounds = compute_fidelity_bounds(
        depth=res_trace.report.surrogate_depth,
        n_features=len(feature_names),
        n_samples=res_trace.report.num_samples,
        empirical_infidelity=1.0 - res_trace.report.fidelity,
    )

    # 2. TRACE — sparse_oblique_tree surrogate (phantom-guided)
    sparse_surr = SparseObliqueTreeSurrogate(
        max_depth=DEPTH,
        min_samples_leaf=5,
        max_iterations=2,
        phantom_threshold=0.3,
        n_probes=20,
        noise_scale=0.05,
        random_state=seed,
    )
    sparse_surr.fit(X_arr, y_bb, model=model, feature_names=tuple(feature_names))
    fid_sparse = accuracy_score(y_bb, sparse_surr.predict(X_arr))
    n_phantom = len(sparse_surr.phantom_features_)
    n_pairs_sparse = len(sparse_surr.interaction_pairs_)

    # 3. Single DT (sklearn, no TRACE wrapper)
    dt = DecisionTreeClassifier(max_depth=DEPTH, random_state=seed)
    dt.fit(X_arr, y_bb)
    dt_fid = accuracy_score(y_bb, dt.predict(X_arr))
    dt_R = dt.get_n_leaves()
    dt_L = dt.get_depth()

    # 4. Anchors (optional)
    anch_fid, anch_R, anch_L = _run_anchors(X_arr, model, feature_names)

    # 5. CORELS (optional)
    corels_fid, corels_R, corels_L = _run_corels(X_arr, y_bb, feature_names)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        # TRACE — decision_tree
        "TRACE_fid": res_trace.report.fidelity,
        "TRACE_R": res_trace.report.num_rules,
        "TRACE_L": res_trace.report.avg_rule_length,
        "TRACE_min_fidelity_pac": bounds.min_fidelity_pac,
        "TRACE_vc_dimension": bounds.vc_dimension,
        # TRACE — sparse_oblique_tree
        "SPARSE_fid": fid_sparse,
        "SPARSE_n_phantom": n_phantom,
        "SPARSE_n_pairs": n_pairs_sparse,
        "SPARSE_n_iterations": sparse_surr.n_iterations_,
        "SPARSE_phantom_features": list(sparse_surr.phantom_features_),
        # Single DT
        "DT_fid": dt_fid,
        "DT_R": dt_R,
        "DT_L": dt_L,
        # Anchors
        "Anchor_fid": anch_fid,
        "Anchor_R": anch_R,
        "Anchor_L": anch_L,
        # CORELS
        "CORELS_fid": corels_fid,
        "CORELS_R": corels_R,
        "CORELS_L": corels_L,
    }


def run_baseline():
    tasks = [
        {"dataset_name": ds, "model_name": m, "seed": s}
        for ds in DATASETS
        for m in MODELS
        for s in SEEDS
    ]
    print(f"Baseline: {len(tasks)} experiments to run")
    results = run_parallel(_run_one, tasks, desc="Baseline")
    save_results(results, "baseline_results.json")

    # Print summary table
    print("\n" + "=" * 95)
    print("  Baseline Summary")
    print("=" * 95)
    print(f"  {'Dataset':<15} {'Model':<6} {'TRACE':>8} {'Sparse':>8} {'DT':>8} "
          f"{'Phantom':>8} {'Pairs':>6} {'PAC LB':>8}")
    print("  " + "-" * 78)
    for r in sorted(results, key=lambda x: (x.get("dataset", ""), x.get("model", ""))):
        if not isinstance(r, dict) or "error" in r:
            continue
        print(f"  {r['dataset']:<15} {r['model']:<6} "
              f"{r['TRACE_fid']:>8.4f} {r['SPARSE_fid']:>8.4f} {r['DT_fid']:>8.4f} "
              f"{r['SPARSE_n_phantom']:>8} {r['SPARSE_n_pairs']:>6} "
              f"{r['TRACE_min_fidelity_pac']:>8.4f}")
    print("=" * 95)


if __name__ == "__main__":
    run_baseline()
