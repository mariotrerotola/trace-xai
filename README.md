# TRACE — Tree-based Rule Approximation for Comprehensible Explanations

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-217%20passed-brightgreen.svg)]()

**TRACE** (`trace-xai`) is a model-agnostic explainability framework that
extracts human-readable IF-THEN rules from *any* classification model
through surrogate decision-tree approximation.

Unlike instance-level methods (LIME, SHAP), TRACE produces a **global, symbolic
explanation** of the entire decision boundary, complemented by rigorous
statistical validation: hold-out fidelity, cross-validated fidelity, bootstrap
confidence intervals, rule-stability analysis, counterfactual boundary
validation, and information-theoretic rule selection.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Model-agnostic** | Works with any object exposing `.predict()` — sklearn, XGBoost, LightGBM, PyTorch wrappers, etc. |
| **Classification** | Uses decision-tree surrogates with accuracy-based fidelity metrics. |
| **Hold-out Fidelity** | Evaluate surrogate faithfulness on unseen data via `X_val` or `validation_split`. |
| **Cross-Validated Fidelity** | *k*-fold CV fidelity with per-fold reports. |
| **Bootstrap Stability** | Jaccard-based rule stability across bootstrap resamples. |
| **Confidence Intervals** | Percentile bootstrap CIs for fidelity and accuracy. |
| **Normalised Complexity** | `avg_conditions_per_feature` and `interaction_strength` metrics. |
| **Regulatory Pruning** | Post-hoc rule simplification: confidence/sample filtering, redundant condition removal, condition truncation. |
| **Monotonicity Constraints** | Enforce domain-consistent rules (e.g. more income → less risk) during fitting and validate post-hoc. |
| **Ensemble Rule Extraction** | Bagging of surrogate trees with fuzzy signature matching for stable, audit-ready rules. |
| **Data Augmentation** | Query synthesis via perturbation, boundary, and sparse-region strategies (inspired by TREPAN). |
| **Oblique Tree Surrogate** | Pairwise feature interaction products for oblique decision boundaries. |
| **Sparse Oblique Tree** | Phantom-guided oblique tree: adds interaction features only for features involved in phantom splits. |
| **Categorical Decoding** | Automatic translation of one-hot/ordinal-encoded conditions back to human-readable category names. |
| **Hyperparameter Presets** | Named presets (`interpretable`, `balanced`, `faithful`) and automated depth selection. |
| **Structural Stability** | Prediction agreement, feature importance rank stability, and top-k feature agreement across bootstraps. |
| **Complementary Metrics** | Boundary agreement, counterfactual consistency, effective complexity, and per-class fidelity. |
| **Theoretical Fidelity Bounds** | PAC-learning, VC-dimension, and Rademacher complexity bounds on surrogate fidelity. |
| **Counterfactual Rule Scoring** | Per-rule validation of whether surrogate split boundaries correspond to real black-box decision boundaries. |
| **MDL Rule Selection** | Information-theoretic rule selection via the Minimum Description Length principle. |
| **Interactive HTML Export** | Self-contained HTML report with filterable, sortable, searchable rules. |
| **Visualization** | matplotlib tree plots and Graphviz DOT export. |
| **Adaptive Tolerance** | Per-feature fuzzy matching tolerances based on feature standard deviation. |
| **Extensible Surrogates** | Protocol-based surrogate interface; decision-tree, oblique-tree, and sparse-oblique-tree implemented. |

---

## Installation

```bash
pip install trace-xai
```

With optional dependencies:

```bash
pip install trace-xai[graphviz]     # Graphviz DOT export
pip install trace-xai[benchmark]    # LIME/SHAP comparative benchmark
pip install trace-xai[dev]          # pytest, xgboost, lightgbm
```

---

## Quick Start

### Classification

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from trace_xai import Explainer

# Train any black-box model
iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# Extract rules
explainer = Explainer(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
)
result = explainer.extract_rules(iris.data, y=iris.target, max_depth=4)

# Inspect
print(result)
# Rule 1: IF petal_length <= 2.4500 THEN class = setosa  [confidence=100.00%, samples=50]
# Rule 2: IF petal_length > 2.4500 AND petal_width <= 1.7500 ...
# ...
# === Fidelity Report ===
#   Fidelity (surrogate vs black-box): 0.9800
#   ...
```

---

## Counterfactual Rule Scoring

Validate whether the surrogate's decision boundaries correspond to real
black-box transitions. Rules whose splits are "phantom" (the black-box
doesn't actually change prediction at that threshold) receive low scores:

```python
# Integrated into extract_rules
result = explainer.extract_rules(
    X, y=y, max_depth=5,
    counterfactual_validity_threshold=0.3,  # filter rules with score < 0.3
)
print(result.counterfactual_report)

# Or as a standalone post-hoc analysis
report = explainer.score_rules_counterfactual(result, X, validity_threshold=0.5)
for rs in report.rule_scores:
    print(f"Rule {rs.rule_index}: score={rs.score:.2f} ({rs.n_valid_conditions}/{rs.n_conditions})")
```

---

## MDL Rule Selection

Select the optimal subset of rules using the Minimum Description Length
principle. Minimises L(model) + L(data|model) — the total cost of encoding
the ruleset structure plus the cost of encoding misclassifications:

```python
# Integrated into extract_rules
result = explainer.extract_rules(
    X, y=y, max_depth=5,
    mdl_selection="forward",  # or "backward", "score_only"
)
print(result.mdl_report)
# === MDL Selection Report ===
#   Rules: 16 -> 10
#   Total MDL: 1245.32 -> 892.10 bits

# Combine with counterfactual scoring
result = explainer.extract_rules(
    X, y=y, max_depth=5,
    counterfactual_validity_threshold=0.3,  # first filter by validity
    mdl_selection="forward",                 # then select by MDL
)
```

---

## Data Augmentation

Expand the training set with synthetic samples to improve surrogate fidelity,
especially in under-represented regions:

```python
result = explainer.extract_rules(
    X, y=y, max_depth=5,
    augmentation="combined",  # "perturbation", "boundary", "sparse", or "combined"
)
```

---

## Hyperparameter Presets

Use named presets for common use cases instead of tuning hyperparameters manually:

```python
# Interpretable: shallow tree (depth=3), few rules
result = explainer.extract_rules(X, y=y, preset="interpretable")

# Balanced: moderate depth (depth=5)
result = explainer.extract_rules(X, y=y, preset="balanced")

# Faithful: deep tree (depth=8), maximum fidelity
result = explainer.extract_rules(X, y=y, preset="faithful")

# Automated depth selection
auto = explainer.auto_select_depth(X, y=y, target_fidelity=0.90)
print(f"Optimal depth: {auto.best_depth}, fidelity: {auto.selected_fidelity:.4f}")
```

---

## Sparse Oblique Tree Surrogate

When the black-box has diagonal decision boundaries, standard axis-aligned
surrogates introduce "phantom splits" — boundaries that don't correspond to
real model transitions. The `SparseObliqueTreeSurrogate` detects these phantoms
via counterfactual probing and adds interaction features only where needed:

```python
from trace_xai import SparseObliqueTreeSurrogate

surrogate = SparseObliqueTreeSurrogate(
    max_depth=5,
    max_iterations=2,
    phantom_threshold=0.3,
    n_probes=20,
)
surrogate.fit(X, y_bb, model=model)

print(f"Phantom features: {surrogate.phantom_features_}")
print(f"Interaction pairs: {surrogate.interaction_pairs_}")
```

---

## Hold-out Fidelity

By default, fidelity is computed **in-sample** (same data used for training and
evaluation). For rigorous assessment, provide a separate validation set or
request an automatic split:

```python
# Option A: explicit validation set
result = explainer.extract_rules(
    X_train, y=y_train,
    X_val=X_test, y_val=y_test,
)
print(result.report.evaluation_type)   # "hold_out"
print(result.train_report)             # in-sample metrics

# Option B: automatic internal split (30% held out)
result = explainer.extract_rules(X, y=y, validation_split=0.3)
print(result.report.evaluation_type)   # "validation_split"
```

---

## Cross-Validated Fidelity

```python
cv_report = explainer.cross_validate_fidelity(X, y=y, n_folds=5)
print(cv_report)
# === Cross-Validated Fidelity (5-fold) ===
#   Mean fidelity: 0.9640 ± 0.0085
#   Mean accuracy:  0.9587 ± 0.0112
print(cv_report.fold_reports[0])  # detailed per-fold report
```

---

## Bootstrap Stability

Measures how consistent the extracted rules are across bootstrap resamples using
pairwise Jaccard similarity:

```python
stability = explainer.compute_stability(X, n_bootstraps=20)
print(stability)
# === Stability Report (20 bootstraps) ===
#   Mean Jaccard: 0.7234 ± 0.0891
```

---

## Confidence Intervals

Bootstrap percentile confidence intervals for fidelity and accuracy, computed as
a separate step to avoid slowing down `extract_rules`:

```python
cis = explainer.compute_confidence_intervals(result, X, y=y, n_bootstraps=1000)
print(f"Fidelity 95% CI: [{cis['fidelity'].lower:.4f}, {cis['fidelity'].upper:.4f}]")
print(f"Accuracy 95% CI: [{cis['accuracy'].lower:.4f}, {cis['accuracy'].upper:.4f}]")
```

---

## Structural Stability

Beyond Jaccard, measure prediction agreement and feature importance rank
stability across bootstrap surrogates:

```python
ss = explainer.compute_structural_stability(X, n_bootstraps=20)
print(f"Prediction agreement: {ss.mean_prediction_agreement:.4f}")
print(f"Feature rank stability: {ss.feature_importance_stability:.4f}")
```

---

## Complementary Metrics

Metrics beyond standard fidelity: boundary agreement, counterfactual
consistency, effective complexity, and per-class fidelity:

```python
metrics = explainer.compute_complementary_metrics(result, X)
print(f"Boundary agreement: {metrics.boundary_agreement:.4f}")
print(f"Counterfactual consistency: {metrics.counterfactual_consistency:.4f}")
print(f"Effective complexity: {metrics.effective_complexity:.4f}")
```

---

## Theoretical Fidelity Bounds

PAC-learning and Rademacher complexity bounds on surrogate fidelity:

```python
bounds = explainer.compute_fidelity_bounds(result)
print(f"VC dimension: {bounds.vc_dimension}")
print(f"Min fidelity (PAC): {bounds.min_fidelity_pac:.4f}")
print(f"Min fidelity (Rademacher): {bounds.min_fidelity_rademacher:.4f}")
```

---

## Visualization

### matplotlib

```python
result.plot(save_path="tree.png", figsize=(24, 12), dpi=200)
```

### Graphviz DOT

```python
dot_string = result.to_dot()
# render with: dot -Tpdf tree.dot -o tree.pdf
```

### Interactive HTML

```python
result.to_html("report.html")
# Opens a self-contained HTML file with:
# - Filterable rules by class/prediction
# - Sortable columns (confidence, samples)
# - Full-text search on feature names
```

---

## API Reference

### `Explainer`

```python
Explainer(model, feature_names, class_names=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | object | Any model with `.predict(X)`. |
| `feature_names` | Sequence[str] | Feature names matching `X.shape[1]`. |
| `class_names` | Sequence[str] or None | Class names for classification. |

#### `Explainer.extract_rules()`

```python
extract_rules(X, *, y=None, max_depth=5, min_samples_leaf=5,
              ccp_alpha=0.0, monotonic_constraints=None, pruning=None,
              X_val=None, y_val=None, validation_split=None,
              surrogate_type="decision_tree",
              augmentation=None, augmentation_kwargs=None,
              preset=None,
              counterfactual_validity_threshold=None,
              counterfactual_noise_scale=0.01,
              mdl_selection=None,
              mdl_precision_bits="auto") -> ExplanationResult
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `X` | — | Training data. |
| `y` | None | True labels (for accuracy metrics only). |
| `max_depth` | 5 | Surrogate tree depth. |
| `min_samples_leaf` | 5 | Min samples per leaf. |
| `ccp_alpha` | 0.0 | Cost-complexity pruning for the sklearn tree. |
| `monotonic_constraints` | None | Feature-name → direction (`+1`/`-1`/`0`) map. |
| `pruning` | None | `PruningConfig` for post-hoc rule pruning. |
| `X_val` | None | External validation set. |
| `y_val` | None | Labels for validation set. |
| `validation_split` | None | Internal split ratio (0-1). Mutually exclusive with `X_val`. |
| `surrogate_type` | `"decision_tree"` | `"decision_tree"` or `"oblique_tree"`. |
| `augmentation` | None | `"perturbation"`, `"boundary"`, `"sparse"`, or `"combined"`. |
| `augmentation_kwargs` | None | Extra kwargs for the augmentation function. |
| `preset` | None | `"interpretable"`, `"balanced"`, or `"faithful"`. |
| `counterfactual_validity_threshold` | None | Filter rules with counterfactual score below this (0-1). |
| `counterfactual_noise_scale` | 0.01 | Perturbation scale for counterfactual boundary probing. |
| `mdl_selection` | None | `"forward"`, `"backward"`, or `"score_only"`. |
| `mdl_precision_bits` | `"auto"` | Bits per threshold for MDL model cost. `"auto"` calibrates from data. |

#### `Explainer.extract_stable_rules()`

```python
extract_stable_rules(X, *, y=None, n_estimators=20, frequency_threshold=0.5,
                     tolerance=0.01, max_depth=5, min_samples_leaf=5,
                     ccp_alpha=0.0, monotonic_constraints=None,
                     random_state=42, X_val=None, y_val=None) -> ExplanationResult
```

#### `Explainer.cross_validate_fidelity()`

```python
cross_validate_fidelity(X, *, y=None, n_folds=5, max_depth=5,
                        min_samples_leaf=5, random_state=42) -> CVFidelityReport
```

#### `Explainer.compute_stability()`

```python
compute_stability(X, *, n_bootstraps=20, max_depth=5,
                  min_samples_leaf=5, random_state=42,
                  tolerance=None) -> StabilityReport
```

#### `Explainer.prune_rules()`

```python
prune_rules(result, config) -> ExplanationResult
```

#### `Explainer.compute_confidence_intervals()`

```python
compute_confidence_intervals(result, X, *, y=None, n_bootstraps=1000,
                             confidence_level=0.95,
                             random_state=42) -> dict[str, ConfidenceInterval]
```

#### `Explainer.score_rules_counterfactual()`

```python
score_rules_counterfactual(result, X, *, validity_threshold=None,
                           noise_scale=0.01,
                           random_state=42) -> CounterfactualReport
```

#### `Explainer.select_rules_mdl()`

```python
select_rules_mdl(result, X, *, method="forward",
                 precision_bits="auto") -> MDLSelectionReport
```

#### `Explainer.auto_select_depth()`

```python
auto_select_depth(X, *, y=None, target_fidelity=0.85, min_depth=2,
                  max_depth=10, n_folds=5, min_samples_leaf=5,
                  random_state=42) -> AutoDepthResult
```

#### `Explainer.compute_structural_stability()`

```python
compute_structural_stability(X, *, n_bootstraps=20, max_depth=5,
                             min_samples_leaf=5, top_k=3,
                             random_state=42) -> StructuralStabilityReport
```

#### `Explainer.compute_complementary_metrics()`

```python
compute_complementary_metrics(result, X) -> ComplementaryMetrics
```

#### `Explainer.compute_fidelity_bounds()`

```python
compute_fidelity_bounds(result, *, delta=0.05) -> FidelityBound
```

#### `Explainer.sensitivity_analysis()`

```python
sensitivity_analysis(X, *, y=None, depth_range=(3,5,7),
                     min_samples_leaf_range=(5,10,20),
                     n_folds=3, random_state=42) -> SensitivityResult
```

### Data Classes

| Class | Key Fields |
|-------|------------|
| `ExplanationResult` | `rules`, `report`, `surrogate`, `train_report`, `pruned_rules`, `pruning_report`, `monotonicity_report`, `ensemble_report`, `stable_rules`, `counterfactual_report`, `mdl_report` |
| `RuleSet` | `rules`, `num_rules`, `avg_conditions`, `max_conditions`, `avg_conditions_per_feature`, `interaction_strength` |
| `Rule` | `conditions`, `prediction`, `confidence`, `samples` |
| `Condition` | `feature`, `operator`, `threshold` |
| `FidelityReport` | `fidelity`, `accuracy`, `evaluation_type`, `fidelity_ci`, `accuracy_ci`, ... |
| `CVFidelityReport` | `mean_fidelity`, `std_fidelity`, `fold_reports`, `n_folds` |
| `StabilityReport` | `mean_jaccard`, `std_jaccard`, `pairwise_jaccards`, `n_bootstraps` |
| `ConfidenceInterval` | `lower`, `upper`, `point_estimate`, `confidence_level` |
| `PruningConfig` | `min_confidence`, `min_samples`, `min_samples_fraction`, `max_conditions`, `remove_redundant` |
| `PruningReport` | `original_count`, `pruned_count`, `removed_low_confidence`, `removed_low_samples`, `conditions_simplified` |
| `MonotonicityReport` | `constraints`, `violations`, `is_compliant` |
| `MonotonicityViolation` | `rule_index`, `rule`, `feature`, `expected_direction`, `description` |
| `MonotonicityEnforcementResult` | `original_rules`, `filtered_rules`, `fidelity_before`, `fidelity_after` |
| `EnsembleReport` | `n_estimators`, `frequency_threshold`, `total_unique_rules`, `stable_rule_count`, `tolerance` |
| `StableRule` | `rule`, `frequency`, `signature`, `variant_count` |
| `HyperparamPreset` | `name`, `max_depth`, `min_samples_leaf`, `ccp_alpha`, `description` |
| `AutoDepthResult` | `best_depth`, `fidelity_scores`, `selected_fidelity`, `target_fidelity` |
| `SensitivityResult` | `results`, `best_config`, `best_fidelity` |
| `StructuralStabilityReport` | `mean_coverage_overlap`, `std_coverage_overlap`, `mean_prediction_agreement`, `std_prediction_agreement`, `feature_importance_stability`, `top_k_feature_agreement`, `n_bootstraps` |
| `ComplementaryMetrics` | `rule_coverage`, `boundary_agreement`, `counterfactual_consistency`, `class_balance_fidelity`, `effective_complexity` |
| `FidelityBound` | `vc_dimension`, `min_fidelity_pac`, `min_fidelity_rademacher`, `sample_complexity_required` |
| `CounterfactualReport` | `rule_scores`, `filtered_ruleset`, `validity_threshold`, `mean_score`, `std_score` |
| `RuleCounterfactualScore` | `rule`, `rule_index`, `score`, `n_conditions`, `n_valid_conditions` |
| `ConditionValidity` | `condition`, `is_valid`, `bb_changes`, `delta_below`, `delta_above` |
| `MDLSelectionReport` | `rule_scores`, `selected_ruleset`, `selection_method`, `total_mdl_before`, `total_mdl_after` |
| `RuleMDLScore` | `rule`, `rule_index`, `model_cost`, `data_cost`, `total_mdl`, `coverage`, `error_rate` |
| `CategoricalMapping` | `original_name`, `encoding`, `encoded_columns`, `categories` |
| `CategoricalCondition` | `feature`, `operator`, `threshold`, `display_value` |

---

## Comparison with Other Methods

| | **TRACE** | LIME | SHAP |
|---|---|---|---|
| **Scope** | Global | Local (per-instance) | Local / Global |
| **Output format** | Symbolic IF-THEN rules | Feature weights | Shapley values |
| **Human readability** | Direct logical rules | Feature importance bars | Feature importance bars |
| **Fidelity metric** | Built-in (+ hold-out, CV, CI) | N/A | N/A |
| **Stability analysis** | Built-in (Jaccard + structural) | N/A | N/A |
| **Counterfactual scoring** | Built-in (per-rule boundary validation) | N/A | N/A |
| **MDL rule selection** | Built-in (information-theoretic) | N/A | N/A |
| **Theoretical bounds** | Built-in (PAC/VC/Rademacher) | N/A | N/A |
| **Model requirement** | `.predict()` | `.predict_proba()` | Model-specific |
| **Speed** | Single tree fit | *n* optimisations | Model-dependent |

TRACE is **complementary** to LIME/SHAP: use TRACE for global rule-based
explanation, LIME/SHAP for local per-instance attribution. The included
`benchmarks/benchmark.py` script directly compares all three.

---

## Project Structure

```
trace-xai/
├── src/trace_xai/
│   ├── __init__.py            # Public API exports
│   ├── explainer.py           # Explainer, ExplanationResult
│   ├── ruleset.py             # Condition, Rule, RuleSet
│   ├── report.py              # FidelityReport, CVFidelityReport, StabilityReport, ConfidenceInterval
│   ├── pruning.py             # PruningConfig, PruningReport, prune_ruleset
│   ├── monotonicity.py        # MonotonicityReport, validate_monotonicity, enforce_monotonicity
│   ├── ensemble.py            # EnsembleReport, StableRule, extract_ensemble_rules
│   ├── augmentation.py        # Data augmentation (perturbation, boundary, sparse region)
│   ├── categorical.py         # Categorical feature decoding (one-hot, ordinal)
│   ├── hyperparams.py         # Presets, auto_select_depth, sensitivity_analysis
│   ├── stability.py           # Structural stability (prediction agreement, feature rank)
│   ├── metrics.py             # Boundary agreement, counterfactual consistency, effective complexity
│   ├── theoretical_bounds.py  # PAC/VC/Rademacher fidelity bounds
│   ├── counterfactual.py      # Counterfactual-guided rule scoring
│   ├── mdl_selection.py       # MDL-based rule selection
│   ├── visualization.py       # plot_surrogate_tree, export_dot
│   ├── html_export.py         # Interactive HTML export
│   └── surrogates/            # Pluggable surrogate backends
│       ├── __init__.py
│       ├── decision_tree.py   # Standard axis-aligned decision tree (default)
│       ├── oblique_tree.py    # Oblique tree (all pairwise feature products)
│       ├── sparse_oblique_tree.py  # Phantom-guided sparse oblique tree
│       ├── rule_list.py       # Placeholder
│       └── gam.py             # Placeholder
├── tests/
│   ├── test_explainer.py      # Core + hold-out + CV + stability + CI tests
│   ├── test_ruleset.py        # Rule data classes + complexity metrics
│   ├── test_report.py         # Fidelity report + bootstrap CI tests
│   ├── test_html_export.py    # HTML export tests
│   ├── test_pruning.py        # Regulatory pruning tests
│   ├── test_monotonicity.py   # Monotonicity constraint tests
│   ├── test_ensemble.py       # Ensemble rule extraction tests
│   ├── test_new_features.py   # Augmentation, presets, structural stability, etc.
│   ├── test_augmentation.py   # Data augmentation tests
│   ├── test_categorical.py    # Categorical decoding tests
│   ├── test_theoretical_bounds.py  # PAC/VC/Rademacher tests
│   ├── test_counterfactual.py # Counterfactual scoring tests
│   └── test_mdl_selection.py  # MDL selection tests
├── experiments/
│   ├── run_rq1.py             # Depth vs fidelity/complexity
│   ├── run_rq2.py             # Rule stability
│   ├── run_rq3.py             # Ensemble extraction
│   ├── run_rq4.py             # Monotonicity constraints
│   ├── run_rq5.py             # Scalability
│   └── run_baseline.py        # Baseline comparison
└── docs/
    ├── getting_started.md     # Installation & first steps
    ├── user_guide.md          # Complete feature guide
    ├── api_reference.md       # Full API documentation
    └── methodology.md         # Scientific methodology & references
```

---

## Running the Benchmark

```bash
pip install trace-xai[benchmark]
python benchmarks/benchmark.py
```

Output compares TRACE vs. LIME vs. SHAP on Iris, Wine, and Breast Cancer
datasets, measuring execution time, fidelity, and feature overlap (Jaccard).

---

## Development

```bash
git clone https://github.com/mariotrerotola/trace-xai.git
cd trace-xai
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Citation

If you use TRACE in your research, please cite:

```bibtex
@software{trace2026,
  title   = {{TRACE}: Tree-based Rule Approximation for Comprehensible Explanations},
  author  = {Trerotola, Mario},
  year    = {2026},
  url     = {https://github.com/mariotrerotola/trace-xai},
  note    = {Python package version 0.2.0},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
