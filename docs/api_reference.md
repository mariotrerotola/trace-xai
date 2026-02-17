# API Reference

Complete reference for all public classes and functions in `trace_xai`.

---

## Module: `trace_xai`

### Public Exports

```python
from trace_xai import (
    # Core
    Explainer,
    ExplanationResult,
    # Rules
    Condition,
    Rule,
    RuleSet,
    # Reports
    FidelityReport,
    CVFidelityReport,
    StabilityReport,
    ConfidenceInterval,
    # Functions
    compute_fidelity_report,
    compute_regression_fidelity_report,
    compute_bootstrap_ci,
    # Visualization
    plot_surrogate_tree,
    export_dot,
    export_html,
)
```

---

## `Explainer`

```python
class Explainer(model, feature_names, class_names=None, *, task=None)
```

Model-agnostic rule extractor. Wraps any model with a `.predict()` method.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | object | *(required)* | Any model exposing a callable `.predict(X)` method. |
| `feature_names` | `Sequence[str]` | *(required)* | Human-readable feature names. Length must match `X.shape[1]`. |
| `class_names` | `Sequence[str]` or `None` | `None` | Class names for classification. If omitted and `task` is not set, regression is assumed. |
| `task` | `str` or `None` | `None` | `"classification"`, `"regression"`, or `None` (auto-detect from `class_names`). |

### Raises

- `TypeError` — if `model` does not have a callable `.predict()` method.
- `ValueError` — if `task` is not one of the accepted values.

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `model` | object | The wrapped black-box model. |
| `feature_names` | `tuple[str, ...]` | Stored feature names. |
| `class_names` | `tuple[str, ...] \| None` | Stored class names (`None` for regression). |

---

### `Explainer.extract_rules()`

```python
def extract_rules(
    self,
    X,
    *,
    y=None,
    max_depth=5,
    min_samples_leaf=5,
    X_val=None,
    y_val=None,
    validation_split=None,
    surrogate_type="decision_tree",
) -> ExplanationResult
```

Extract interpretable IF-THEN rules from the black-box model.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `X` | array-like `(n, p)` | *(required)* | Training / explanation data. |
| `y` | array-like `(n,)` | `None` | True labels. Used only for accuracy metrics in the report. |
| `max_depth` | `int` | `5` | Maximum depth of the surrogate tree. |
| `min_samples_leaf` | `int` | `5` | Minimum samples per leaf in the surrogate tree. |
| `X_val` | array-like | `None` | Separate validation set for hold-out fidelity evaluation. |
| `y_val` | array-like | `None` | True labels for the validation set. |
| `validation_split` | `float` | `None` | Internal split ratio (0 < value < 1). Mutually exclusive with `X_val`. |
| `surrogate_type` | `str` | `"decision_tree"` | Surrogate backend. Only `"decision_tree"` is currently supported. |

#### Returns

`ExplanationResult`

#### Raises

- `ValueError` — if both `X_val` and `validation_split` are provided.
- `ValueError` — if `surrogate_type` is not `"decision_tree"`.

---

### `Explainer.cross_validate_fidelity()`

```python
def cross_validate_fidelity(
    self,
    X,
    *,
    y=None,
    n_folds=5,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
) -> CVFidelityReport
```

Perform *k*-fold cross-validated fidelity estimation. For each fold, a fresh
surrogate is trained on the training split and evaluated on the held-out split.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `X` | array-like `(n, p)` | *(required)* | Feature matrix. |
| `y` | array-like `(n,)` | `None` | True labels (enables accuracy reporting per fold). |
| `n_folds` | `int` | `5` | Number of cross-validation folds. |
| `max_depth` | `int` | `5` | Surrogate tree depth. |
| `min_samples_leaf` | `int` | `5` | Minimum samples per leaf. |
| `random_state` | `int` | `42` | Random seed for reproducibility. |

#### Returns

`CVFidelityReport`

---

### `Explainer.compute_stability()`

```python
def compute_stability(
    self,
    X,
    *,
    n_bootstraps=20,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
) -> StabilityReport
```

Compute rule stability via bootstrap resampling. Generates `n_bootstraps`
bootstrap samples of `X`, extracts rules from each, and computes all pairwise
Jaccard similarities on canonical rule signatures.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `X` | array-like `(n, p)` | *(required)* | Feature matrix. |
| `n_bootstraps` | `int` | `20` | Number of bootstrap resamples. |
| `max_depth` | `int` | `5` | Surrogate tree depth. |
| `min_samples_leaf` | `int` | `5` | Minimum samples per leaf. |
| `random_state` | `int` | `42` | Random seed. |

#### Returns

`StabilityReport`

---

### `Explainer.compute_confidence_intervals()`

```python
def compute_confidence_intervals(
    self,
    result,
    X,
    *,
    y=None,
    n_bootstraps=1000,
    confidence_level=0.95,
    random_state=42,
) -> dict[str, ConfidenceInterval]
```

Compute bootstrap percentile confidence intervals for fidelity (and accuracy
if `y` is provided).

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `ExplanationResult` | *(required)* | A previously computed explanation result (provides the fitted surrogate). |
| `X` | array-like `(n, p)` | *(required)* | Feature matrix (same or different from training). |
| `y` | array-like `(n,)` | `None` | True labels. If provided, an `"accuracy"` CI is also computed. |
| `n_bootstraps` | `int` | `1000` | Number of bootstrap iterations. |
| `confidence_level` | `float` | `0.95` | Confidence level (e.g. 0.95 for 95% CI). |
| `random_state` | `int` | `42` | Random seed. |

#### Returns

`dict[str, ConfidenceInterval]` with keys `"fidelity"` and (optionally)
`"accuracy"`.

---

## `ExplanationResult`

```python
class ExplanationResult
```

Container returned by `Explainer.extract_rules()`.

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `rules` | `RuleSet` | The extracted IF-THEN rules. |
| `report` | `FidelityReport` | Fidelity metrics (evaluated on validation data if hold-out was used). |
| `surrogate` | `DecisionTreeClassifier` or `DecisionTreeRegressor` | The fitted surrogate tree. |
| `train_report` | `FidelityReport \| None` | In-sample report (only when hold-out evaluation was used). |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `plot(*, save_path=None, **kwargs)` | `None` | Render the surrogate tree via matplotlib. |
| `to_dot()` | `str` | Export the surrogate tree as a Graphviz DOT string. |
| `to_html(output_path)` | `None` | Write an interactive HTML report. |
| `__str__()` | `str` | Human-readable rules + fidelity report. |

---

## `Condition`

```python
@dataclass(frozen=True)
class Condition
```

A single split condition.

| Field | Type | Description |
|-------|------|-------------|
| `feature` | `str` | Feature name. |
| `operator` | `str` | `"<="` or `">"`. |
| `threshold` | `float` | Split threshold value. |

`str(Condition("age", "<=", 30.5))` → `"age <= 30.5000"`

---

## `Rule`

```python
@dataclass(frozen=True)
class Rule
```

An IF-THEN rule extracted from a decision-tree leaf.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conditions` | `tuple[Condition, ...]` | — | Conjunction of conditions. |
| `prediction` | `str` | — | Predicted class name (or formatted value for regression). |
| `samples` | `int` | — | Number of training samples at this leaf. |
| `confidence` | `float` | — | Dominant-class fraction (classification) or 1.0 (regression). |
| `leaf_id` | `int` | — | Node id in the surrogate tree. |
| `prediction_value` | `float \| None` | `None` | Numeric prediction (regression only). |

**String representation:**

- Classification: `IF age <= 30.5000 AND income > 50000.0000 THEN class = approved  [confidence=92.00%, samples=46]`
- Regression: `IF temperature <= 25.3000 THEN value = 142.5678  [samples=38]`

---

## `RuleSet`

```python
@dataclass(frozen=True)
class RuleSet
```

An ordered collection of rules extracted from a surrogate tree.

| Field / Property | Type | Description |
|-----------------|------|-------------|
| `rules` | `tuple[Rule, ...]` | Ordered rules. |
| `feature_names` | `tuple[str, ...]` | Feature names. |
| `class_names` | `tuple[str, ...]` | Class names. |
| `num_rules` | `int` | Number of rules (property). |
| `avg_conditions` | `float` | Mean conditions per rule (property). |
| `max_conditions` | `int` | Maximum conditions in any rule (property). |
| `avg_conditions_per_feature` | `float` | `avg_conditions / len(feature_names)` (property). |
| `interaction_strength` | `float` | Fraction of rules with > 1 distinct feature (property). |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `filter_by_class(class_name)` | `RuleSet` | Subset of rules predicting `class_name`. |
| `rule_signatures()` | `frozenset[str]` | Canonical rule strings (for Jaccard comparison). |
| `to_text()` | `str` | Human-readable multi-line string. |

---

## `FidelityReport`

```python
@dataclass(frozen=True)
class FidelityReport
```

Quantitative summary of surrogate faithfulness.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fidelity` | `float` | — | Agreement between surrogate and black-box (0–1). |
| `accuracy` | `float \| None` | — | Surrogate accuracy vs true labels. |
| `blackbox_accuracy` | `float \| None` | — | Black-box accuracy vs true labels. |
| `num_rules` | `int` | — | Number of rules (leaves). |
| `avg_rule_length` | `float` | — | Mean conditions per rule. |
| `max_rule_length` | `int` | — | Max conditions per rule. |
| `surrogate_depth` | `int` | — | Fitted surrogate tree depth. |
| `surrogate_n_leaves` | `int` | — | Number of leaves. |
| `num_samples` | `int` | — | Samples used for evaluation. |
| `class_fidelity` | `dict[str, float]` | — | Per-class fidelity. |
| `evaluation_type` | `str` | `"in_sample"` | `"in_sample"`, `"hold_out"`, `"validation_split"`, or `"cross_validation"`. |
| `avg_conditions_per_feature` | `float \| None` | `None` | Normalised complexity. |
| `interaction_strength` | `float \| None` | `None` | Multi-feature interaction metric. |
| `fidelity_ci` | `ConfidenceInterval \| None` | `None` | Bootstrap CI for fidelity. |
| `accuracy_ci` | `ConfidenceInterval \| None` | `None` | Bootstrap CI for accuracy. |
| `fidelity_r2` | `float \| None` | `None` | R² surrogate vs black-box (regression). |
| `fidelity_mse` | `float \| None` | `None` | MSE surrogate vs black-box (regression). |
| `accuracy_r2` | `float \| None` | `None` | R² surrogate vs true labels (regression). |
| `accuracy_mse` | `float \| None` | `None` | MSE surrogate vs true labels (regression). |

---

## `CVFidelityReport`

```python
@dataclass(frozen=True)
class CVFidelityReport
```

| Field | Type | Description |
|-------|------|-------------|
| `mean_fidelity` | `float` | Mean fidelity across folds. |
| `std_fidelity` | `float` | Standard deviation of fidelity. |
| `mean_accuracy` | `float \| None` | Mean accuracy (if `y` was provided). |
| `std_accuracy` | `float \| None` | Std of accuracy. |
| `fold_reports` | `list[FidelityReport]` | Per-fold detailed reports. |
| `n_folds` | `int` | Number of folds. |

---

## `StabilityReport`

```python
@dataclass(frozen=True)
class StabilityReport
```

| Field | Type | Description |
|-------|------|-------------|
| `mean_jaccard` | `float` | Mean pairwise Jaccard similarity. |
| `std_jaccard` | `float` | Std of pairwise Jaccard. |
| `pairwise_jaccards` | `list[float]` | All `C(n,2)` pairwise values. |
| `n_bootstraps` | `int` | Number of bootstrap resamples. |

---

## `ConfidenceInterval`

```python
@dataclass(frozen=True)
class ConfidenceInterval
```

| Field | Type | Description |
|-------|------|-------------|
| `lower` | `float` | Lower bound. |
| `upper` | `float` | Upper bound. |
| `point_estimate` | `float` | Mean of bootstrap distribution. |
| `confidence_level` | `float` | Confidence level (e.g. 0.95). |

---

## Standalone Functions

### `compute_fidelity_report()`

```python
compute_fidelity_report(surrogate, X, y_bb, y_true, class_names,
                        num_rules, avg_rule_length, max_rule_length,
                        *, evaluation_type="in_sample",
                        avg_conditions_per_feature=None,
                        interaction_strength=None) -> FidelityReport
```

Compute classification fidelity metrics from a fitted surrogate.

### `compute_regression_fidelity_report()`

```python
compute_regression_fidelity_report(surrogate, X, y_bb, y_true,
                                   num_rules, avg_rule_length, max_rule_length,
                                   *, evaluation_type="in_sample",
                                   avg_conditions_per_feature=None,
                                   interaction_strength=None) -> FidelityReport
```

Compute regression fidelity metrics (R², MSE) from a fitted surrogate.

### `compute_bootstrap_ci()`

```python
compute_bootstrap_ci(values, *, confidence_level=0.95) -> ConfidenceInterval
```

Compute a percentile bootstrap CI from an array of resampled metric values.

### `plot_surrogate_tree()`

```python
plot_surrogate_tree(surrogate, feature_names, class_names=(),
                    *, figsize=(20,10), fontsize=9,
                    save_path=None, dpi=150) -> None
```

### `export_dot()`

```python
export_dot(surrogate, feature_names, class_names=()) -> str
```

### `export_html()`

```python
export_html(result, output_path) -> None
```

Write a self-contained HTML file with interactive rule table.
