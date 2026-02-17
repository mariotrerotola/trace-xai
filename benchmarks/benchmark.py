#!/usr/bin/env python3
"""Benchmark: compare trace_xai with LIME and SHAP.

Measures fidelity, feature overlap, and wall-clock time on standard
sklearn datasets (Iris, Wine, Breast Cancer).

Install dependencies:
    pip install trace_xai[benchmark]
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer


def _load_datasets() -> List[Dict[str, Any]]:
    loaders = [load_iris, load_wine, load_breast_cancer]
    datasets = []
    for loader in loaders:
        ds = loader()
        datasets.append({
            "name": ds.get("DESCR", "").split("\n")[0].strip(". ") or loader.__name__,
            "X": ds.data,
            "y": ds.target,
            "feature_names": list(ds.feature_names),
            "class_names": list(ds.target_names),
        })
    return datasets


def _benchmark_trace_xai(
    model, X, feature_names, class_names,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    exp = Explainer(model, feature_names=feature_names, class_names=class_names)
    result = exp.extract_rules(X, max_depth=5)
    elapsed = time.perf_counter() - t0
    features_used = set()
    for rule in result.rules.rules:
        for cond in rule.conditions:
            features_used.add(cond.feature)
    return {
        "method": "trace_xai",
        "time_s": elapsed,
        "fidelity": result.report.fidelity,
        "n_features_used": len(features_used),
        "features": features_used,
    }


def _benchmark_lime(model, X, feature_names, class_names) -> Dict[str, Any]:
    try:
        import lime.lime_tabular
    except ImportError:
        return {"method": "LIME", "error": "lime not installed"}

    t0 = time.perf_counter()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X, feature_names=feature_names, class_names=class_names, mode="classification",
    )
    features_used = set()
    n_samples = min(50, len(X))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X), size=n_samples, replace=False)
    for idx in indices:
        exp = explainer.explain_instance(X[idx], model.predict_proba, num_features=5)
        for feat_name, _ in exp.as_list():
            clean = feat_name.split(" ")[0] if " " in feat_name else feat_name
            features_used.add(clean)
    elapsed = time.perf_counter() - t0
    return {
        "method": "LIME",
        "time_s": elapsed,
        "n_features_used": len(features_used),
        "features": features_used,
    }


def _benchmark_shap(model, X, feature_names) -> Dict[str, Any]:
    try:
        import shap
    except ImportError:
        return {"method": "SHAP", "error": "shap not installed"}

    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    elapsed = time.perf_counter() - t0
    if isinstance(shap_values, list):
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importance = np.abs(shap_values).mean(axis=0)
    top_k = min(5, len(feature_names))
    top_indices = np.argsort(importance)[-top_k:]
    features_used = {feature_names[i] for i in top_indices}
    return {
        "method": "SHAP",
        "time_s": elapsed,
        "n_features_used": len(features_used),
        "features": features_used,
    }


def run_benchmark() -> None:
    """Run the full benchmark suite and print results."""
    datasets = _load_datasets()

    for ds in datasets:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {ds['name']} ({ds['X'].shape[0]} samples, "
              f"{ds['X'].shape[1]} features, {len(ds['class_names'])} classes)")
        print(f"{'=' * 70}")

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(ds["X"], ds["y"])

        results = []
        results.append(_benchmark_trace_xai(
            model, ds["X"], ds["feature_names"], ds["class_names"],
        ))
        results.append(_benchmark_lime(
            model, ds["X"], ds["feature_names"], ds["class_names"],
        ))
        results.append(_benchmark_shap(model, ds["X"], ds["feature_names"]))

        print(f"\n  {'Method':<22} {'Time (s)':>10} {'Fidelity':>10} {'#Features':>10}")
        print(f"  {'─' * 54}")
        for r in results:
            if "error" in r:
                print(f"  {r['method']:<22} {'N/A':>10} {'N/A':>10} {r['error']}")
                continue
            fid = f"{r['fidelity']:.4f}" if "fidelity" in r else "N/A"
            print(f"  {r['method']:<22} {r['time_s']:>10.4f} {fid:>10} {r['n_features_used']:>10}")

        # Feature overlap
        eg_feats = results[0].get("features", set())
        for r in results[1:]:
            if "error" in r:
                continue
            other_feats = r.get("features", set())
            overlap = eg_feats & other_feats
            union = eg_feats | other_feats
            jaccard = len(overlap) / len(union) if union else 0
            print(f"\n  Feature overlap (trace_xai vs {r['method']}): "
                  f"Jaccard={jaccard:.2f}, shared={overlap}")


if __name__ == "__main__":
    run_benchmark()
