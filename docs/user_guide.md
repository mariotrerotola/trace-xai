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
9. [Visualization](#9-visualization)
10. [Interactive HTML Export](#10-interactive-html-export)
11. [Working with Rules Programmatically](#11-working-with-rules-programmatically)
12. [Surrogate Backends](#12-surrogate-backends)
13. [Benchmarking Against LIME and SHAP](#13-benchmarking-against-lime-and-shap)

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

## 9. Visualization

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

## 10. Interactive HTML Export

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

## 11. Working with Rules Programmatically

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

## 12. Surrogate Backends

TRACE uses a pluggable surrogate architecture defined by the `BaseSurrogate`
protocol in `trace_xai.surrogates.base`:

```python
from trace_xai.surrogates import BaseSurrogate, DecisionTreeSurrogate
```

Currently implemented:

| Backend | Status | Module |
|---------|--------|--------|
| `DecisionTreeSurrogate` | Implemented | `surrogates.decision_tree` |
| `RuleListSurrogate` | Placeholder (raises `NotImplementedError`) | `surrogates.rule_list` |
| `GAMSurrogate` | Placeholder (raises `NotImplementedError`) | `surrogates.gam` |

The `surrogate_type` parameter in `extract_rules()` currently only accepts
`"decision_tree"`. Future versions will support `"rule_list"` and `"gam"`.

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

## 13. Benchmarking Against LIME and SHAP

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
