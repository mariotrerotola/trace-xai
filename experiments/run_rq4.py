"""RQ4: Effect of monotonic constraints on rule extraction.

Parallelises over (dataset, model) combinations.
New: uses enforce_monotonicity for automated constraint enforcement.
"""

import numpy as np
from experiments.datasets import load_dataset
from experiments.models import get_model
from experiments.utils import run_parallel, save_results
from trace_xai import Explainer, enforce_monotonicity, validate_monotonicity

# Configuration
DATASETS = ["adult", "german_credit"]
MODELS = ["rf", "xgb", "mlp"]
SEEDS = [42]
DEPTH = 4

CONSTRAINTS = {
    "adult": {"age": 1, "hours-per-week": 1, "education-num": 1},
    "german_credit": {"credit_amount": -1, "duration": -1, "age": 1},
}


def _run_one(dataset_name, model_name, seed):
    """Single (dataset, model) — unconstrained vs constrained vs enforced."""
    X, y, task, feature_names, _ = load_dataset(dataset_name)
    if X is None:
        return {"dataset": dataset_name, "model": model_name, "seed": seed,
                "error": "dataset_load_failed"}

    constraints = CONSTRAINTS.get(dataset_name, {})
    if not constraints:
        return {"dataset": dataset_name, "model": model_name, "seed": seed,
                "error": "no_constraints"}

    model = get_model(model_name, task, random_state=seed)
    model.fit(X, y)
    explainer = Explainer(model, feature_names=feature_names, task=task)

    # 1. Unconstrained
    res_unc = explainer.extract_rules(X, max_depth=DEPTH)
    report_unc = validate_monotonicity(res_unc.rules, constraints)
    fid_unc = res_unc.report.fidelity
    viol_unc = len(report_unc.violations)

    # 2. With surrogate constraints (if sklearn supports monotonic_cst)
    fid_con, viol_con = None, None
    try:
        res_con = explainer.extract_rules(X, max_depth=DEPTH,
                                          monotonic_constraints=constraints)
        fid_con = res_con.report.fidelity
        viol_con = (len(res_con.monotonicity_report.violations)
                    if res_con.monotonicity_report else 0)
    except Exception:
        pass

    # 3. Post-hoc enforcement (new feature)
    enforcement = enforce_monotonicity(res_unc.rules, constraints)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        # Unconstrained
        "fidelity_unconstrained": fid_unc,
        "violations_unconstrained": viol_unc,
        "rules_unconstrained": res_unc.rules.num_rules,
        # Constrained (surrogate-level)
        "fidelity_constrained": fid_con,
        "violations_constrained": viol_con,
        # Post-hoc enforcement (new)
        "rules_after_enforcement": enforcement.corrected_ruleset.num_rules,
        "rules_removed": enforcement.rules_removed,
        "fidelity_impact": enforcement.fidelity_impact,
    }


def run_rq4():
    tasks = [
        {"dataset_name": ds, "model_name": m, "seed": s}
        for ds in DATASETS
        for m in MODELS
        for s in SEEDS
    ]
    print(f"RQ4: {len(tasks)} experiments to run")
    results = run_parallel(_run_one, tasks, desc="RQ4")
    save_results(results, "rq4_results.json")


if __name__ == "__main__":
    run_rq4()
