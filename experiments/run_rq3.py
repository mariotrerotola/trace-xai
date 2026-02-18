"""RQ3: Ensemble stability — adaptive vs fixed frequency thresholds.

Parallelises over (dataset, model, phi) combinations.
Uses extract_ensemble_rules_adaptive and rank_rules_by_frequency.
"""

from collections import Counter

import numpy as np
import pandas as pd
from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results
from trace_xai import Explainer, extract_ensemble_rules_adaptive, rank_rules_by_frequency
from trace_xai.ensemble import extract_ensemble_rules
from trace_xai.ruleset import RuleSet

# Configuration
DATASETS = ["adult", "german_credit", "compas"]
MODELS = ["rf", "xgb", "mlp"]
N_ESTIMATORS = 50
PHIS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEEDS = [42]
DEPTH = 4


def _build_bootstrap_rulesets(explainer, X, seed):
    """Generate bootstrap rulesets (shared across phi values)."""
    X_arr = np.asarray(X)
    rng = np.random.RandomState(seed)
    rulesets = []

    for _ in range(N_ESTIMATORS):
        idx = rng.choice(len(X_arr), size=len(X_arr), replace=True)
        X_boot = X_arr[idx]
        y_bb_boot = np.asarray(explainer.model.predict(X_boot))

        surr = explainer._build_surrogate(max_depth=DEPTH,
                                          min_samples_leaf=5, random_state=seed)
        surr.fit(X_boot, y_bb_boot)
        rules = explainer._extract_rules_from_tree(surr)
        rulesets.append(RuleSet(
            rules=tuple(rules),
            feature_names=tuple(explainer.feature_names),
            class_names=explainer.class_names or (),
        ))

    return rulesets


def _predict_with_rules(X, stable_rules, default_prediction, feature_names):
    """Vectorised rule-based prediction."""
    if isinstance(X, pd.DataFrame):
        X_df = X
    else:
        X_df = pd.DataFrame(X, columns=feature_names)

    n = len(X_df)
    parsed = []
    for sr in stable_rules:
        r = sr.rule
        conds = [(c.feature, c.operator, c.threshold) for c in r.conditions]
        try:
            pred_val = float(r.prediction)
        except (ValueError, TypeError):
            pred_val = 0.0
        parsed.append((conds, pred_val))

    # Build match matrix
    matches = np.zeros((len(parsed), n), dtype=bool)
    for i, (conds, _) in enumerate(parsed):
        mask = np.ones(n, dtype=bool)
        for feat, op, thresh in conds:
            if feat not in X_df.columns:
                continue
            col = X_df[feat].values
            if op == "<=":
                mask &= col <= thresh
            else:
                mask &= col > thresh
        matches[i] = mask

    preds = np.full(n, default_prediction, dtype=float)
    matches_T = matches.T
    for j in range(n):
        idxs = np.where(matches_T[j])[0]
        if len(idxs) == 0:
            continue
        vals = [parsed[k][1] for k in idxs]
        c = Counter(vals)
        preds[j] = c.most_common(1)[0][0]

    return preds


def _run_one(dataset_name, model_name, seed):
    """One (dataset, model) — evaluates ALL phi values + adaptive."""
    X, y, task, feature_names, _ = load_dataset(dataset_name)
    if X is None:
        return []

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    model = get_model(model_name, task, random_state=seed)
    model.fit(X, y)
    explainer = Explainer(model, feature_names=feature_names, task=task)

    rulesets = _build_bootstrap_rulesets(explainer, X, seed)

    y_bb = np.asarray(model.predict(X_arr))
    default_pred = float(np.argmax(np.bincount(y_arr.astype(int))))

    results = []

    from sklearn.metrics import accuracy_score

    # Fixed-phi experiments
    for phi in PHIS:
        stable_rules, report = extract_ensemble_rules(
            rulesets, frequency_threshold=phi, tolerance=0.01,
        )
        preds = _predict_with_rules(X, stable_rules, default_pred, feature_names)
        fid = accuracy_score(y_bb, preds)

        results.append({
            "dataset": dataset_name, "model": model_name, "seed": seed,
            "method": "fixed_phi", "phi": phi,
            "num_stable_rules": len(stable_rules), "fidelity": fid,
        })

    # Adaptive phi
    adaptive_rules, adaptive_report = extract_ensemble_rules_adaptive(
        rulesets, tolerance=0.01, min_rules=3,
    )
    preds_a = _predict_with_rules(X, adaptive_rules, default_pred, feature_names)
    fid_a = accuracy_score(y_bb, preds_a)

    results.append({
        "dataset": dataset_name, "model": model_name, "seed": seed,
        "method": "adaptive_phi",
        "phi": adaptive_report.frequency_threshold,
        "num_stable_rules": len(adaptive_rules), "fidelity": fid_a,
    })

    # Ranked rules (soft filtering)
    ranked = rank_rules_by_frequency(rulesets, tolerance=0.01)
    results.append({
        "dataset": dataset_name, "model": model_name, "seed": seed,
        "method": "ranked",
        "phi": None,
        "num_stable_rules": len(ranked),
        "fidelity": None,
        "top5_frequencies": [r.frequency for r in ranked[:5]],
    })

    return results


def run_rq3():
    tasks = [
        {"dataset_name": ds, "model_name": m, "seed": s}
        for ds in DATASETS
        for m in MODELS
        for s in SEEDS
    ]
    print(f"RQ3: {len(tasks)} experiment groups to run")
    results = run_parallel(_run_one, tasks, desc="RQ3")
    save_results(results, "rq3_results.json")


if __name__ == "__main__":
    run_rq3()
