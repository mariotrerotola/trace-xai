"""RQ6–RQ7: Counterfactual scoring and MDL-based rule selection across all datasets.

Evaluates the CF+MDL pipeline with sparse_oblique_tree surrogate on all benchmark
datasets and 3 black-box models.  Reports:
  - Surrogate type used (sparse_oblique_tree)
  - Phantom features detected and interaction pairs added
  - Raw rules extracted (after pruning)
  - CF validity scores (mean, per-rule breakdown)
  - Rules retained after CF filtering
  - Rules retained after MDL forward selection
  - MDL compression (bits saved)
  - Fidelity delta: sparse vs decision_tree baseline

Parallelises over (dataset, model) combinations.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results

from trace_xai import (
    Explainer,
    PruningConfig,
    SparseObliqueTreeSurrogate,
    auto_select_depth,
    augment_data,
    score_rules_counterfactual,
    select_rules_mdl,
)

# Configuration
DATASETS = ["adult", "german_credit", "compas"]
MODELS = ["rf", "xgb", "mlp"]
SEEDS = [42]
TARGET_FIDELITY = 0.85
CF_THRESHOLD = 0.3
CF_NOISE = 0.1
CF_N_PROBES = 30
MDL_PRECISION_BITS = 8


def _run_one(dataset_name, model_name, seed):
    """Run CF+MDL pipeline with sparse_oblique_tree on a single (dataset, model) pair."""
    X, y, task, feature_names, _ = load_dataset(dataset_name)
    if X is None:
        return []

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=seed, stratify=y_arr,
    )

    # Train black-box
    model = get_model(model_name, task, random_state=seed)
    model.fit(X_train, y_train)

    bb_acc = accuracy_score(y_test, model.predict(X_test))
    y_bb_train = np.asarray(model.predict(X_train))

    # Create explainer
    n_classes = len(np.unique(y_arr))
    class_names = [str(c) for c in range(n_classes)]
    explainer = Explainer(model, feature_names=feature_names,
                          class_names=class_names, task=task)

    # ── Baseline: decision_tree fidelity (for delta comparison) ──────────
    res_dt = explainer.extract_rules(X_train, max_depth=4,
                                     surrogate_type="decision_tree")
    fid_dt_baseline = res_dt.report.fidelity

    # ── sparse_oblique_tree: standalone for phantom analysis ──────────────
    sparse_probe = SparseObliqueTreeSurrogate(
        max_depth=4,
        min_samples_leaf=5,
        max_iterations=2,
        phantom_threshold=0.3,
        n_probes=15,
        noise_scale=0.05,
        random_state=seed,
    )
    sparse_probe.fit(X_train, y_bb_train, model=model,
                     feature_names=tuple(feature_names))
    fid_sparse_standalone = accuracy_score(y_bb_train, sparse_probe.predict(X_train))

    # Auto-select depth
    depth_result = auto_select_depth(
        explainer, X_train, y=y_train,
        target_fidelity=TARGET_FIDELITY,
        min_depth=3, max_depth=6,
        n_folds=3,
        random_state=seed,
    )
    best_depth = depth_result.best_depth

    # Augment (subsample for large datasets)
    rng = np.random.RandomState(seed)
    n_aug = min(3000, len(X_train))
    aug_idx = rng.choice(len(X_train), size=n_aug, replace=False)
    X_aug, y_aug = augment_data(
        X_train[aug_idx], model,
        strategy="perturbation",
        n_neighbors=2,
        noise_scale=0.05,
        random_state=seed,
    )

    # ── Extract rules: sparse_oblique_tree + pruning ──────────────────────
    result_raw = explainer.extract_rules(
        X_aug, y=y_aug,
        max_depth=best_depth,
        surrogate_type="sparse_oblique_tree",
        X_val=X_test, y_val=y_test,
        pruning=PruningConfig(
            min_confidence=0.55,
            min_samples=10,
            remove_redundant=True,
        ),
    )

    raw_n_rules = result_raw.rules.num_rules
    raw_fidelity = result_raw.report.fidelity

    # Inspect the sparse surrogate from the result
    sparse_surr = result_raw.surrogate
    n_phantom = len(sparse_surr.phantom_features_)
    n_pairs = len(sparse_surr.interaction_pairs_)
    n_iter = sparse_surr.n_iterations_

    # ── Counterfactual scoring ────────────────────────────────────────────
    cf_report = score_rules_counterfactual(
        result_raw.rules, model, X_test,
        validity_threshold=CF_THRESHOLD,
        noise_scale=CF_NOISE,
        n_probes=CF_N_PROBES,
        random_state=seed,
    )

    cf_mean = cf_report.mean_score
    cf_std = cf_report.std_score
    cf_retained = cf_report.n_rules_retained
    cf_per_rule = [rs.score for rs in cf_report.rule_scores]

    if cf_report.filtered_ruleset and cf_report.filtered_ruleset.num_rules > 0:
        ruleset_after_cf = cf_report.filtered_ruleset
    else:
        ruleset_after_cf = result_raw.rules

    # ── MDL selection ─────────────────────────────────────────────────────
    mdl_report = select_rules_mdl(
        ruleset_after_cf, model, X_test,
        n_classes=n_classes,
        method="forward",
        precision_bits=MDL_PRECISION_BITS,
    )

    mdl_selected = mdl_report.n_rules_selected
    mdl_reduction = mdl_report.mdl_reduction

    if mdl_report.selected_ruleset.num_rules > 0:
        final_rules = mdl_report.selected_ruleset
    else:
        final_rules = ruleset_after_cf

    final_n_rules = final_rules.num_rules

    # Score only for per-rule error rates
    mdl_score_only = select_rules_mdl(
        final_rules, model, X_test,
        n_classes=n_classes,
        method="score_only",
    )
    avg_error_rate = (
        float(np.mean([rs.error_rate for rs in mdl_score_only.rule_scores]))
        if mdl_score_only.rule_scores else None
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "task": task,
        "bb_accuracy": float(bb_acc),
        "best_depth": best_depth,
        # Phantom analysis
        "n_phantom_features": n_phantom,
        "n_interaction_pairs": n_pairs,
        "n_sparse_iterations": n_iter,
        # Fidelity comparison
        "fid_dt_baseline": float(fid_dt_baseline),
        "fid_sparse_standalone": float(fid_sparse_standalone),
        "fid_delta": float(fid_sparse_standalone - fid_dt_baseline),
        # Pipeline results
        "raw_rules": raw_n_rules,
        "raw_fidelity": float(raw_fidelity),
        "cf_mean": float(cf_mean),
        "cf_std": float(cf_std),
        "cf_retained": cf_retained,
        "cf_per_rule": cf_per_rule,
        "mdl_selected": mdl_selected,
        "mdl_reduction_bits": float(mdl_reduction),
        "final_rules": final_n_rules,
        "avg_error_rate": avg_error_rate,
    }


def run_rq6_rq7():
    tasks = [
        {"dataset_name": ds, "model_name": m, "seed": s}
        for ds in DATASETS
        for m in MODELS
        for s in SEEDS
    ]
    print(f"RQ6-RQ7: {len(tasks)} experiments to run")
    results = run_parallel(_run_one, tasks, desc="RQ6-RQ7")
    save_results(results, "rq6_rq7_results.json")

    # Print summary table
    print("\n" + "=" * 105)
    print("  RQ6-RQ7 Summary: sparse_oblique_tree + Counterfactual + MDL Pipeline")
    print("=" * 105)
    print(f"  {'Dataset':<15} {'Model':<6} {'Raw':>5} {'CF↓':>5} {'MDL↓':>5} "
          f"{'CF μ':>6} {'Fid(S)':>7} {'ΔFid':>7} {'Phantom':>8} {'Pairs':>6} "
          f"{'Err%':>6} {'Bits↓':>7}")
    print("  " + "-" * 95)

    for r in sorted(results, key=lambda x: (x.get("dataset", ""), x.get("model", ""))):
        if not isinstance(r, dict):
            continue
        err = r.get("avg_error_rate")
        err_str = f"{err:>6.2f}" if err is not None else "   N/A"
        delta = r.get("fid_delta", 0.0)
        delta_str = f"{delta:>+7.4f}"
        print(f"  {r.get('dataset',''):<15} {r.get('model',''):<6} "
              f"{r.get('raw_rules',''):>5} {r.get('cf_retained',''):>5} "
              f"{r.get('final_rules',''):>5} "
              f"{r.get('cf_mean', 0):>6.2f} "
              f"{r.get('fid_sparse_standalone', 0):>7.4f} "
              f"{delta_str} "
              f"{r.get('n_phantom_features', 0):>8} "
              f"{r.get('n_interaction_pairs', 0):>6} "
              f"{err_str} "
              f"{r.get('mdl_reduction_bits', 0):>7.1f}")

    print("=" * 105)


if __name__ == "__main__":
    run_rq6_rq7()
