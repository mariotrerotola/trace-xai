# User Guide

This guide covers every feature of TRACE in depth, with executable examples.

---

## Table of Contents

1. [Core Workflow](#1-core-workflow)
2. [Classification](#2-classification)
3. [Regression](#3-regression)
4. [Hold-out Fidelity Evaluation](#4-hold-out-fidelity-evaluation)
5. [Cross-Validated Fidelity](#5-cross-validated-fidelity)
6. [Bootstrap Stability](#6-bootstrap-stability)
7. [Confidence Intervals](#7-confidence-intervals)
8. [Normalised Complexity Metrics](#8-normalised-complexity-metrics)
9. [Regulatory Pruning](#9-regulatory-pruning)
10. [Monotonicity Constraints](#10-monotonicity-constraints)
11. [Ensemble Rule Extraction](#11-ensemble-rule-extraction)
12. [Data Augmentation](#12-data-augmentation)
13. [Categorical Feature Decoding](#13-categorical-feature-decoding)
14. [Hyperparameter Presets & Auto-Depth](#14-hyperparameter-presets--auto-depth)
15. [Structural Stability](#15-structural-stability)
16. [Complementary Metrics](#16-complementary-metrics)
17. [Theoretical Fidelity Bounds](#17-theoretical-fidelity-bounds)
18. [Counterfactual Rule Scoring](#18-counterfactual-rule-scoring)
19. [MDL Rule Selection](#19-mdl-rule-selection)
20. [Visualization](#20-visualization)
21. [Interactive HTML Export](#21-interactive-html-export)
22. [Working with Rules Programmatically](#22-working-with-rules-programmatically)
23. [Surrogate Backends](#23-surrogate-backends)
24. [Benchmarking Against LIME and SHAP](#24-benchmarking-against-lime-and-shap)
25. [Adaptive Tolerance](#25-adaptive-tolerance)

---

## 1. Core Workflow

TRACE follows a three-step pipeline:

```
Black-box model
      │
      ▼
┌──────────────┐    predict(X)     ┌────────────────┐
│  Your Model  │ ─────────────────▶│  Predictions   │
│  (any .predict)                  │  y_bb          │
└──────────────┘                   └───────┬────────┘
                                           │
                                           ▼
                                   ┌────────────────┐
                                   │  Surrogate DT  │  ← trained on (X, y_bb)
                                   │  (shallow tree) │
                                   └───────┬────────┘
                                           │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                         IF-THEN       Fidelity    Tree
                          Rules        Report      Plot
```

1. The black-box model produces predictions on the explanation data.
2. A shallow decision tree (the *surrogate*) is trained to mimic those
   predictions.
3. Rules are extracted from the surrogate's leaves; fidelity metrics quantify
   how faithful the surrogate is to the original model.

---

## 2. Classification

```python
from sklearn.ensemble import GradientBoostingClassifier
from trace_xai import Explainer

model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

explainer = Explainer(
    model,
    feature_names=["age", "income", "score"],
    class_names=["rejected", "approved"],
)

result = explainer.extract_rules(X_train, y=y_train, max_depth=5)
```

`class_names` triggers classification mode automatically. The surrogate uses
`DecisionTreeClassifier`, and fidelity is measured via accuracy (agreement rate).

### Controlling surrogate complexity

| Parameter | Effect |
|-----------|--------|
| `max_depth=3` | Shallower tree, fewer rules, lower fidelity. |
| `max_depth=8` | Deeper tree, more rules, higher fidelity. |
| `min_samples_leaf=10` | Avoids tiny leaves, increases generalisation. |

**Trade-off:** deeper surrogates are more faithful but less interpretable. A
typical workflow is to try `max_depth` in {3, 5, 8} and pick the value that
balances fidelity with the number of rules.

---

## 3. Regression

```python
from sklearn.ensemble import RandomForestRegressor
from trace_xai import Explainer

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = Explainer(
    model,
    feature_names=["temperature", "humidity", "wind_speed"],
    # class_names omitted → regression auto-detected
)

result = explainer.extract_rules(X_train, y=y_train, max_depth=5)
```

In regression mode:

- The surrogate is a `DecisionTreeRegressor`.
- Rules show `THEN value = X.XXXX` instead of `THEN class = ...`.
- The report includes `fidelity_r2` (R² between surrogate and black-box) and
  `fidelity_mse` instead of class-based accuracy.
- `accuracy_r2` and `accuracy_mse` compare the surrogate to the true labels.

You can also force the task explicitly:

```python
explainer = Explainer(model, feature_names=feat, task="regression")
```

---

## 4. Hold-out Fidelity Evaluation

In-sample fidelity can be optimistically biased. TRACE offers two ways to
evaluate on unseen data:

### Option A: Provide an external validation set

```python
result = explainer.extract_rules(
    X_train, y=y_train,
    X_val=X_test, y_val=y_test,
)

# result.report → evaluated on X_test (hold_out)
# result.train_report → evaluated on X_train (in_sample)
print(result.report.evaluation_type)        # "hold_out"
print(result.train_report.evaluation_type)  # "in_sample"
```

### Option B: Automatic internal split

```python
result = explainer.extract_rules(
    X, y=y,
    validation_split=0.3,  # 30% held out
)
print(result.report.evaluation_type)  # "validation_split"
```

For classification, the split is stratified. `X_val` and `validation_split` are
mutually exclusive — providing both raises `ValueError`.

---

## 5. Cross-Validated Fidelity

For a robust, low-variance fidelity estimate:

```python
cv_report = explainer.cross_validate_fidelity(
    X, y=y,
    n_folds=5,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
)

print(cv_report)
# === Cross-Validated Fidelity (5-fold) ===
#   Mean fidelity: 0.9640 ± 0.0085
#   Mean accuracy:  0.9587 ± 0.0112

# Access individual fold reports
for i, fold in enumerate(cv_report.fold_reports, 1):
    print(f"Fold {i}: fidelity={fold.fidelity:.4f}, rules={fold.num_rules}")
```

Each fold trains a fresh surrogate on the training split and evaluates on the
held-out split, ensuring no data leakage.

---

## 6. Bootstrap Stability

Rule stability measures how consistent the extracted rules are when the
training data is perturbed:

```python
stability = explainer.compute_stability(
    X,
    n_bootstraps=20,
    max_depth=5,
    random_state=42,
)

print(stability)
# === Stability Report (20 bootstraps) ===
#   Mean Jaccard: 0.7234 ± 0.0891

# All pairwise Jaccard similarities (C(20,2) = 190 values)
print(len(stability.pairwise_jaccards))  # 190
```

**Interpretation:**

| Mean Jaccard | Interpretation |
|-------------|----------------|
| > 0.8 | Highly stable rules |
| 0.5 – 0.8 | Moderately stable |
| < 0.5 | Unstable — consider increasing `min_samples_leaf` or decreasing `max_depth` |

The method generates `n_bootstraps` bootstrap samples, fits a surrogate to
each, extracts rules, and computes pairwise Jaccard similarity on the canonical
rule signatures.

---

## 7. Confidence Intervals

Bootstrap percentile confidence intervals for fidelity and accuracy. This is a
separate method (not computed inside `extract_rules`) because it requires many
bootstrap iterations and would slow down the main extraction:

```python
cis = explainer.compute_confidence_intervals(
    result, X,
    y=y,
    n_bootstraps=1000,
    confidence_level=0.95,
    random_state=42,
)

fid_ci = cis["fidelity"]
print(f"Fidelity: {fid_ci.point_estimate:.4f} "
      f"[{fid_ci.lower:.4f}, {fid_ci.upper:.4f}]")

if "accuracy" in cis:
    acc_ci = cis["accuracy"]
    print(f"Accuracy: {acc_ci.point_estimate:.4f} "
          f"[{acc_ci.lower:.4f}, {acc_ci.upper:.4f}]")
```

The `"accuracy"` key is only present when `y` is provided.

---

## 8. Normalised Complexity Metrics

Standard rule complexity metrics (avg rule length, max rule length) are not
comparable across datasets with different feature counts. TRACE provides:

### `avg_conditions_per_feature`

Average conditions per rule, divided by the number of features. A ruleset with
avg 3 conditions over 100 features (0.03) is simpler than one with avg 3
conditions over 4 features (0.75).

### `interaction_strength`

Fraction of rules that reference more than one distinct feature. Values close to
0 indicate the surrogate uses mostly univariate splits; values close to 1
indicate heavy multi-feature interactions.

```python
report = result.report
print(f"Avg conditions/feature: {report.avg_conditions_per_feature:.4f}")
print(f"Interaction strength:   {report.interaction_strength:.4f}")

# Also available on the RuleSet directly:
print(result.rules.avg_conditions_per_feature)
print(result.rules.interaction_strength)
```

---

## 9. Regulatory Pruning

For regulatory contexts (GDPR, EU AI Act, financial regulators), rules with
many conditions are hard to justify. TRACE provides configurable post-hoc
pruning:

### Inline pruning (during extraction)

```python
from trace_xai import Explainer, PruningConfig

result = explainer.extract_rules(
    X, y=y,
    max_depth=5,
    ccp_alpha=0.01,  # sklearn cost-complexity pruning
    pruning=PruningConfig(
        min_confidence=0.6,       # remove uncertain rules
        min_samples=20,           # remove rare rules
        max_conditions=4,         # truncate long rules
        remove_redundant=True,    # simplify A>5 AND A>3 → A>5
    ),
)

# Original rules are always preserved
print(f"Original: {result.rules.num_rules} rules")
print(f"Pruned:   {result.pruned_rules.num_rules} rules")

# Detailed pruning report
print(result.pruning_report)
```

### Post-hoc pruning (after extraction)

```python
result = explainer.extract_rules(X, y=y)
pruned_result = explainer.prune_rules(
    result,
    PruningConfig(min_confidence=0.7, max_conditions=3),
)
```

### PruningConfig parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_confidence` | `0.0` | Remove rules below this confidence. |
| `min_samples` | `0` | Remove rules covering fewer samples. |
| `min_samples_fraction` | `0.0` | Remove rules covering less than this fraction. Requires `total_samples`. |
| `max_conditions` | `None` | Truncate rules to at most this many conditions. |
| `remove_redundant` | `False` | Simplify redundant conditions on the same feature. |
| `total_samples` | `None` | Total dataset size (needed for `min_samples_fraction`). |

---

## 10. Monotonicity Constraints

Financial regulations require that certain features have a univocal effect
(e.g. higher income should not increase credit risk). TRACE enforces this
both during surrogate fitting and via post-hoc validation:

```python
result = explainer.extract_rules(
    X, y=y,
    monotonic_constraints={
        "income": 1,       # increasing income → lower risk
        "debt_ratio": -1,  # increasing debt → higher risk
        "age": 0,          # no constraint
    },
)

# Check compliance
print(result.monotonicity_report)
# === Monotonicity Report ===
#   Compliant: True
#   Violations: 0
```

### Post-hoc validation and filtering

```python
from trace_xai import validate_monotonicity, filter_monotonic_violations

report = validate_monotonicity(result.rules, {"income": 1, "debt_ratio": -1})
if not report.is_compliant:
    for v in report.violations:
        print(f"  Rule {v.rule_index}: {v.description}")
    # Remove violating rules
    clean_rules = filter_monotonic_violations(result.rules, report)
```

### Constraint values

| Value | Meaning |
|-------|---------|
| `+1` | Prediction must increase (or stay same) as feature increases. |
| `-1` | Prediction must decrease (or stay same) as feature increases. |
| `0` | No constraint. |

> **Note:** The `monotonic_cst` sklearn parameter requires scikit-learn >= 1.3.
> TRACE checks at runtime and raises a clear error if not available.

---

## 11. Ensemble Rule Extraction

A single surrogate tree can produce different rules with slight data changes.
For auditable processes, use ensemble extraction to find **stable rules**
that appear consistently across multiple bootstrap surrogates:

```python
result = explainer.extract_stable_rules(
    X, y=y,
    n_estimators=30,           # number of bootstrap surrogates
    frequency_threshold=0.5,   # rule must appear in >= 50% of trees
    tolerance=0.1,             # fuzzy threshold matching
    max_depth=5,
)

# Ensemble report
print(result.ensemble_report)
# === Ensemble Report (30 surrogates) ===
#   Stable rules: 8
#   Total unique rules: 45
#   Mean rules per tree: 12.3

# Access stable rules with frequency information
for sr in result.stable_rules:
    print(f"  [{sr.frequency:.0%}] {sr.rule}")
```

### Fuzzy stability analysis

The existing `compute_stability()` can also use fuzzy matching for more
realistic Jaccard scores:

```python
# Exact matching (original behavior)
exact = explainer.compute_stability(X, n_bootstraps=20)

# Fuzzy matching (near-identical thresholds are considered the same rule)
fuzzy = explainer.compute_stability(X, n_bootstraps=20, tolerance=0.1)
print(f"Exact Jaccard: {exact.mean_jaccard:.4f}")
print(f"Fuzzy Jaccard: {fuzzy.mean_jaccard:.4f}")  # typically higher
```

### Combining all three features

```python
# 1. Extract stable rules with monotonicity
result = explainer.extract_stable_rules(
    X, y=y,
    n_estimators=20,
    frequency_threshold=0.5,
    monotonic_constraints={"income": 1, "debt_ratio": -1},
)

# 2. Prune the stable rules
pruned = explainer.prune_rules(
    result,
    PruningConfig(min_confidence=0.7, max_conditions=4, remove_redundant=True),
)

# 3. Export for audit
pruned.to_html("audit_report.html")
```

---

## 12. Data Augmentation

TRACE can augment the explanation data with synthetic samples to improve
surrogate fidelity, especially in sparse regions of the input space:

```python
from trace_xai import augment_data

# One-call augmentation combining all strategies
X_aug, y_aug = augment_data(
    X_train, model, surrogate,       # surrogate required for "boundary", "sparse", "combined"
    strategy="combined",              # "perturbation", "boundary", "sparse", or "combined"
    n_neighbors=5,                    # perturbation neighbours per sample
    noise_scale=0.1,                  # perturbation noise intensity
    n_boundary_samples=200,           # boundary samples to generate
    n_sparse_samples=100,             # sparse-region samples
    random_state=42,
)

# Use augmented data for extraction
result = explainer.extract_rules(X_aug, max_depth=5)
```

> **Note:** The `surrogate` parameter is required for `"boundary"`, `"sparse"`, and
> `"combined"` strategies (the surrogate's decision boundaries are used to guide
> sampling). For `"perturbation"` only, `surrogate` can be `None`.

### Individual strategies

```python
from trace_xai import (
    perturbation_augmentation,
    boundary_augmentation,
    sparse_region_augmentation,
)

# 1. LIME-style local perturbation (no surrogate needed)
X_pert, y_pert = perturbation_augmentation(
    X_train, model, n_neighbors=5, noise_scale=0.1,
)

# 2. Boundary-focused sampling (requires a fitted surrogate)
X_bound, y_bound = boundary_augmentation(
    X_train, model, surrogate, n_samples=200,
)

# 3. Sparse region filling (requires a fitted surrogate)
X_sparse, y_sparse = sparse_region_augmentation(
    X_train, model, surrogate, n_samples=100,
)
```

---

## 13. Categorical Feature Decoding

Rules on one-hot or ordinal-encoded features (e.g. `occupation_0 <= 0.5`) are
unreadable. TRACE can decode them back to human-readable form:

```python
from trace_xai import CategoricalMapping, decode_ruleset

# Define how each categorical feature was encoded
mappings = [
    CategoricalMapping(
        original_name="occupation",
        encoding="onehot",
        encoded_columns=("occupation_Tech", "occupation_Prof", "occupation_Sales"),
        categories=("Tech", "Prof", "Sales"),
    ),
    CategoricalMapping(
        original_name="education",
        encoding="ordinal",
        encoded_columns=("education",),
        categories=("HS", "Bachelors", "Masters", "PhD"),
    ),
]

# Decode the extracted ruleset
decoded = decode_ruleset(result.rules, mappings)
print(decoded)
# Before: IF occupation_Tech <= 0.5 AND education <= 2.5 THEN ...
# After:  IF occupation ≠ Tech AND education ∈ {HS, Bachelors, Masters} THEN ...
```

---

## 14. Hyperparameter Presets & Auto-Depth

TRACE provides named presets and automatic depth selection to reduce the
manual tuning burden:

### Named presets

```python
from trace_xai import get_preset, PRESETS

# List available presets
for name, preset in PRESETS.items():
    print(f"{name}: max_depth={preset.max_depth}, "
          f"min_samples_leaf={preset.min_samples_leaf}")

# Use a preset
result = explainer.extract_rules(X, y=y, preset="balanced")
```

| Preset | max_depth | min_samples_leaf | ccp_alpha | Description |
|--------|-----------|------------------|-----------|-------------|
| `"interpretable"` | 3 | 20 | 0.01 | Maximum interpretability: shallow trees, few rules, easy to audit. |
| `"balanced"` | 5 | 10 | 0.005 | Balance between fidelity and interpretability. |
| `"faithful"` | 8 | 5 | 0.0 | Maximum fidelity: deeper trees, more rules, closer to black-box. |

### Automatic depth selection

```python
from trace_xai import auto_select_depth

# Find the shallowest depth achieving target fidelity
depth_result = auto_select_depth(
    explainer, X, y=y,
    min_depth=2,
    max_depth=10,
    target_fidelity=0.95,
)

print(f"Selected depth: {depth_result.best_depth}")
print(f"Achieved fidelity: {depth_result.selected_fidelity:.4f}")

# Use the auto-selected depth
result = explainer.extract_rules(X, y=y, max_depth=depth_result.best_depth)
```

### Sensitivity analysis

```python
from trace_xai import sensitivity_analysis

sa = sensitivity_analysis(
    explainer, X, y=y,
    depth_range=[2, 3, 4, 5, 6, 8, 10],
)

for entry in sa.results:
    print(f"depth={entry['max_depth']}: fidelity={entry['mean_fidelity']:.4f}, "
          f"rules={entry['num_rules']}")
```

---

## 15. Structural Stability

Syntactic Jaccard similarity can be misleading (two rulesets covering the
same regions with slightly different thresholds score 0.0). Structural
stability uses semantic metrics:

```python
from trace_xai import compute_structural_stability

report = compute_structural_stability(
    explainer, X,
    n_bootstraps=20,
    max_depth=5,
)

print(report)
# === Structural Stability Report (20 bootstraps) ===
#   Coverage overlap: 0.8723 ± 0.0456
#   Prediction agreement: 0.9512 ± 0.0234
#   Feature importance stability: 0.9234 ± 0.0312
```

| Metric | What It Measures |
|--------|-----------------|
| `coverage_overlap` | Do bootstrap rulesets cover the same samples? |
| `prediction_agreement` | Do bootstrap surrogates agree on predictions? |
| `feature_importance_stability` | Do bootstrap surrogates use the same features? |

---

## 16. Complementary Metrics

These metrics address the circularity criticism (the surrogate is trained and
evaluated on the same fidelity criterion):

```python
from trace_xai import compute_complementary_metrics

metrics = compute_complementary_metrics(
    result.surrogate, model, X, result.rules,
    class_names=("setosa", "versicolor", "virginica"),
)

print(metrics)
# === Complementary Metrics ===
#   Rule coverage: 1.0000
#   Boundary agreement: 0.8456
#   Counterfactual consistency: 0.8912
#   Effective complexity: 0.8571
#   Class-balanced fidelity:
#     setosa: 1.0000
#     versicolor: 0.9500
#     virginica: 0.9800
```

| Metric | Description |
|--------|-------------|
| `rule_coverage` | Fraction of samples covered by at least one rule |
| `boundary_agreement` | Agreement near decision boundaries (hardest region) |
| `counterfactual_consistency` | Consistency of counterfactual changes between surrogate and black-box |
| `class_balance_fidelity` | Per-class fidelity weighted by class prevalence |
| `effective_complexity` | Fraction of rules that activate on test samples |

---

## 17. Theoretical Fidelity Bounds

Compute PAC-learning and Rademacher complexity bounds on how well a depth-D
surrogate can approximate any black-box model:

```python
from trace_xai import compute_fidelity_bounds

bounds = compute_fidelity_bounds(
    depth=5,
    n_features=4,
    n_samples=150,
    empirical_infidelity=0.02,  # 1 - fidelity
    delta=0.05,                 # failure probability (confidence = 1 - delta = 0.95)
)

print(bounds)
# === Fidelity Bounds (depth=5, p=4, N=150) ===
#   VC dimension: 63
#   PAC estimation error: ≤ 0.1234
#   Rademacher estimation error: ≤ 0.0987
#   Guaranteed min fidelity (PAC): ≥ 0.8566
#   Sample complexity for ε=0.05: N ≥ 1234
```

### Utility functions

```python
from trace_xai import (
    vc_dimension_decision_tree,
    sample_complexity,
    relu_network_regions,
    min_depth_for_regions,
    optimal_depth_bound,
)

# VC dimension of depth-5 trees over 4 features
vc = vc_dimension_decision_tree(depth=5, n_features=4)

# Minimum samples for PAC guarantee
n_min = sample_complexity(vc_dim=vc, epsilon=0.05, delta=0.05)

# How many linear regions does a ReLU network have?
regions = relu_network_regions(n_layers=3, width=64, input_dim=4)

# Minimum depth to partition those regions
d_min = min_depth_for_regions(regions)
```

---

## 18. Counterfactual Rule Scoring

Counterfactual scoring verifies whether each rule's split thresholds
correspond to **real** decision boundaries of the black-box model. A high
fidelity surrogate may still place splits at locations the black-box ignores
("phantom splits"). This feature addresses that fundamental limitation.

```python
from trace_xai import score_rules_counterfactual

# Score all rules by counterfactual validity
cf_report = score_rules_counterfactual(
    result.rules, model, X,
    validity_threshold=0.5,  # filter rules with score < 0.5
    noise_scale=0.01,        # perturbation around thresholds
    n_probes=20,             # probe pairs per condition (robustness)
    random_state=42,
)

print(cf_report)
# === Counterfactual Validity Report ===
#   Rules scored: 7
#   Mean validity score: 0.7234 ± 0.1456
#   Validity threshold: 0.50
#   Rules retained: 5/7

# Access filtered ruleset
if cf_report.filtered_ruleset is not None:
    print(f"Retained {cf_report.n_rules_retained} rules")
```

### Inline scoring during extraction

```python
result = explainer.extract_rules(
    X, y=y,
    max_depth=5,
    counterfactual_validity_threshold=0.5,
    counterfactual_noise_scale=0.01,
)

# Report is attached to the result
print(result.counterfactual_report)
```

### How it works

For each condition in a rule (e.g. `petal_length <= 2.45`):

1. Generate `n_probes` random base samples within feature ranges.
2. For each probe, create a paired sample: one just below the threshold,
   one just above.
3. Query the black-box on both samples.
4. If the black-box prediction changes in **at least one probe**, the
   condition is **valid** (it corresponds to a real decision boundary).

Using multiple probes makes the method robust to ensemble models (GBM,
Random Forest) whose decision boundaries are not axis-aligned — a single
probe might miss the boundary depending on the values of other features.

The rule's score is the fraction of valid conditions.

---

## 19. MDL Rule Selection

MDL (Minimum Description Length) provides an information-theoretic criterion
for selecting rules. Instead of frequency-based filtering, rules are selected
by minimising the total description length:

**MDL = L(model) + L(data | model)**

where L(model) is the cost in bits to encode the rule structure and
L(data | model) is the cost to encode misclassifications.

```python
from trace_xai import select_rules_mdl

# Select rules using forward greedy MDL
mdl_report = select_rules_mdl(
    result.rules, model, X,
    n_classes=3,
    precision_bits=16,
    method="forward",       # "forward", "backward", or "score_only"
)

print(mdl_report)
# === MDL Selection Report ===
#   Method: forward
#   Rules: 7 -> 5
#   Total MDL: 234.56 -> 178.23 bits
#   MDL reduction: 56.33 bits

# Access the selected ruleset
selected = mdl_report.selected_ruleset
```

### Inline selection during extraction

```python
result = explainer.extract_rules(
    X, y=y,
    max_depth=5,
    mdl_selection="forward",
    mdl_precision_bits=16,
)

# Report is attached to the result
print(result.mdl_report)
```

### Selection methods

| Method | Strategy |
|--------|----------|
| `"forward"` | Greedily add rules whose MDL cost is less than the null hypothesis cost |
| `"backward"` | Start with all rules, iteratively remove the worst |
| `"score_only"` | Compute MDL scores without filtering |

### Combining counterfactual + MDL

Both features can be chained in a single `extract_rules()` call. Counterfactual
filtering runs first, then MDL selection:

```python
result = explainer.extract_rules(
    X, y=y,
    max_depth=5,
    counterfactual_validity_threshold=0.5,
    mdl_selection="forward",
)

print(f"Original rules: {len(result.rules.rules)}")
print(f"After CF + MDL: rules are filtered and selected")
```

---

## 20. Visualization

### matplotlib tree plot

```python
# Display inline (Jupyter) or save to file
result.plot(save_path="tree.png", figsize=(24, 12), fontsize=10, dpi=200)
```

### Graphviz DOT export

```python
dot_string = result.to_dot()

# Save to file
with open("tree.dot", "w") as f:
    f.write(dot_string)

# Render with graphviz CLI:
#   dot -Tpdf tree.dot -o tree.pdf
#   dot -Tsvg tree.dot -o tree.svg
```

### Programmatic Graphviz rendering

```python
import graphviz
graph = graphviz.Source(result.to_dot())
graph.render("tree", format="pdf")
```

---

## 21. Interactive HTML Export

Generate a self-contained HTML file with no external dependencies:

```python
result.to_html("report.html")
```

The HTML report includes:

- **Fidelity report** rendered at the top.
- **Rule table** with columns: #, Conditions, Prediction, Confidence, Samples.
- **Class filter buttons** — click to show only rules for a specific class.
- **Full-text search** — type a feature name to filter rules.
- **Sortable columns** — click any header to sort (ascending/descending).

The file is fully self-contained (inline CSS and JavaScript) and can be shared,
embedded in reports, or opened in any browser.

---

## 22. Working with Rules Programmatically

### Filtering rules

```python
# Only rules predicting "setosa"
setosa_rules = result.rules.filter_by_class("setosa")
print(f"{setosa_rules.num_rules} rules for setosa")

# Sort by confidence
sorted_rules = sorted(result.rules.rules, key=lambda r: r.confidence, reverse=True)
```

### Accessing conditions

```python
for rule in result.rules.rules:
    for cond in rule.conditions:
        print(f"  Feature: {cond.feature}, Op: {cond.operator}, "
              f"Threshold: {cond.threshold:.4f}")
```

### Rule signatures (for comparison)

```python
sigs = result.rules.rule_signatures()
# frozenset of canonical string representations
```

### Text export

```python
text = result.rules.to_text()
with open("rules.txt", "w") as f:
    f.write(text)
```

---

## 23. Surrogate Backends

TRACE uses a pluggable surrogate architecture defined by the `BaseSurrogate`
protocol in `trace_xai.surrogates.base`:

```python
from trace_xai.surrogates import BaseSurrogate, DecisionTreeSurrogate
```

Currently implemented:

| Backend | Status | Module | Description |
|---------|--------|--------|-------------|
| `DecisionTreeSurrogate` | Implemented | `surrogates.decision_tree` | Standard axis-aligned decision tree (default). |
| `ObliqueTreeSurrogate` | Implemented | `surrogates.oblique_tree` | Augments feature space with all pairwise interaction products for oblique splits. |
| `SparseObliqueTreeSurrogate` | Implemented | `surrogates.sparse_oblique_tree` | Phantom-guided oblique tree: only adds interaction features for features involved in phantom splits. |
| `RuleListSurrogate` | Placeholder | `surrogates.rule_list` | Not yet implemented. |
| `GAMSurrogate` | Placeholder | `surrogates.gam` | Not yet implemented. |

The `surrogate_type` parameter in `extract_rules()` accepts `"decision_tree"`
or `"oblique_tree"`. You can also pass a `SparseObliqueTreeSurrogate` instance
directly for more control.

### Oblique Tree Surrogate

The `ObliqueTreeSurrogate` approximates oblique (diagonal) decision boundaries
by augmenting the feature space with pairwise interaction terms (products).
A split on `feature_A * feature_B <= threshold` corresponds to an oblique
hyperplane in the original space:

```python
result = explainer.extract_rules(X, y=y, max_depth=5, surrogate_type="oblique_tree")
```

### Sparse Oblique Tree Surrogate

The `SparseObliqueTreeSurrogate` addresses the "phantom split" problem: when
the black-box has diagonal decision boundaries, axis-aligned surrogates place
splits where the black-box prediction does not actually change. Instead of
adding all pairwise interactions (which explodes combinatorially), it:

1. Fits an initial axis-aligned tree.
2. Probes each internal node counterfactually to detect phantom splits.
3. Adds interaction features **only** for features involved in phantom splits.
4. Re-fits on the augmented space. Iterates up to `max_iterations`.

```python
from trace_xai import SparseObliqueTreeSurrogate

surrogate = SparseObliqueTreeSurrogate(
    max_depth=5,
    max_iterations=2,         # phantom detection iterations
    phantom_threshold=0.3,    # fraction of probes that must NOT change BB
    noise_scale=0.01,         # perturbation magnitude
    n_probes=20,              # probe pairs per node
    max_interaction_features=None,  # cap on interaction features (None = no cap)
)

# Fit with black-box model for phantom-guided selection
surrogate.fit(X_train, y_bb, model=model)

# Inspect results
print(f"Phantom features: {surrogate.phantom_features_}")
print(f"Interaction pairs: {surrogate.interaction_pairs_}")
print(f"Iterations: {surrogate.n_iterations_}")

# Get augmented feature names
names = surrogate.get_augmented_feature_names(feature_names)
```

When no `model` is provided, the surrogate falls back to importance-based
interaction selection using the top-k most important features from the initial
tree.

### Implementing a custom surrogate

Any object implementing the `BaseSurrogate` protocol can be used:

```python
class MySurrogate:
    def fit(self, X, y): ...
    def predict(self, X): ...
    def get_depth(self) -> int: ...
    def get_n_leaves(self) -> int: ...
```

---

## 24. Benchmarking Against LIME and SHAP

TRACE includes a ready-to-run benchmark script:

```bash
pip install trace-xai[benchmark]
python benchmarks/benchmark.py
```

The benchmark:

1. Trains a `RandomForestClassifier` on each of three datasets (Iris, Wine,
   Breast Cancer).
2. Runs TRACE, LIME (50-instance sample), and SHAP (`TreeExplainer`).
3. Reports execution time, fidelity (TRACE only), number of features used, and
   pairwise feature overlap (Jaccard).

Example output:

```
======================================================================
  Dataset: Iris (150 samples, 4 features, 3 classes)
======================================================================

  Method                   Time (s)   Fidelity  #Features
  ──────────────────────────────────────────────────────────
  trace_xai          0.0042     0.9800          3
  LIME                       2.3456        N/A          4
  SHAP                       0.1234        N/A          4

  Feature overlap (trace_xai vs LIME): Jaccard=0.75, shared={...}
  Feature overlap (trace_xai vs SHAP): Jaccard=0.67, shared={...}
```

---

## 25. Adaptive Tolerance

The fuzzy matching tolerance `delta` used in ensemble rule extraction and
stability analysis is often set as a single global constant. This can be
problematic when features have very different scales.

`compute_adaptive_tolerance` computes per-feature tolerances as a fraction of
each feature's standard deviation:

```python
from trace_xai import compute_adaptive_tolerance

tolerances = compute_adaptive_tolerance(
    X, feature_names=["age", "income", "score"],
    scale=0.05,  # 5% of each feature's std
)
# {'age': 0.7321, 'income': 1523.45, 'score': 0.0412}

# Use with ensemble extraction
result = explainer.extract_stable_rules(
    X, y=y,
    n_estimators=20,
    tolerance=tolerances,  # per-feature adaptive tolerance
)
```

This addresses the criticism that the global tolerance `delta` is an arbitrary
parameter with no principled selection guideline.
