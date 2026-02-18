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
    # Pruning
    PruningConfig,
    PruningReport,
    prune_ruleset,
    # Monotonicity
    MonotonicityReport,
    MonotonicityViolation,
    MonotonicityEnforcementResult,
    validate_monotonicity,
    filter_monotonic_violations,
    enforce_monotonicity,
    # Ensemble
    EnsembleReport,
    StableRule,
    extract_ensemble_rules_adaptive,
    rank_rules_by_frequency,
    # Augmentation
    augment_data,
    perturbation_augmentation,
    boundary_augmentation,
    sparse_region_augmentation,
    # Categorical
    CategoricalMapping,
    CategoricalCondition,
    decode_ruleset,
    decode_conditions,
    # Hyperparameters
    HyperparamPreset,
    AutoDepthResult,
    SensitivityResult,
    get_preset,
    auto_select_depth,
    sensitivity_analysis,
    compute_adaptive_tolerance,
    PRESETS,
    # Structural stability
    StructuralStabilityReport,
    compute_structural_stability,
    # Complementary metrics
    ComplementaryMetrics,
    compute_complementary_metrics,
    # Theoretical bounds
    FidelityBound,
    compute_fidelity_bounds,
    vc_dimension_decision_tree,
    estimation_error_pac,
    estimation_error_rademacher,
    sample_complexity,
    relu_network_regions,
    min_depth_for_regions,
    optimal_depth_bound,
    # Counterfactual scoring
    ConditionValidity,
    RuleCounterfactualScore,
    CounterfactualReport,
    score_rules_counterfactual,
    # MDL selection
    RuleMDLScore,
    MDLSelectionReport,
    select_rules_mdl,
    binary_entropy,
    compute_rule_model_cost,
    # Surrogates
    BaseSurrogate,
    DecisionTreeSurrogate,
    ObliqueTreeSurrogate,
    SparseObliqueTreeSurrogate,
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
    ccp_alpha=0.0,
    monotonic_constraints=None,
    pruning=None,
    X_val=None,
    y_val=None,
    validation_split=None,
    surrogate_type="decision_tree",
    augmentation=None,
    augmentation_kwargs=None,
    preset=None,
    counterfactual_validity_threshold=None,
    counterfactual_noise_scale=0.01,
    mdl_selection=None,
    mdl_precision_bits="auto",
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
| `ccp_alpha` | `float` | `0.0` | Cost-complexity pruning parameter for the sklearn tree. |
| `monotonic_constraints` | `dict[str, int]` | `None` | Map feature names to `+1` (increasing), `-1` (decreasing), or `0` (no constraint). |
| `pruning` | `PruningConfig` | `None` | Post-hoc rule pruning configuration. |
| `X_val` | array-like | `None` | Separate validation set for hold-out fidelity evaluation. |
| `y_val` | array-like | `None` | True labels for the validation set. |
| `validation_split` | `float` | `None` | Internal split ratio (0 < value < 1). Mutually exclusive with `X_val`. |
| `surrogate_type` | `str` | `"decision_tree"` | `"decision_tree"` or `"oblique_tree"`. |
| `augmentation` | `str` | `None` | Data augmentation strategy: `"perturbation"`, `"boundary"`, `"sparse"`, or `"combined"`. |
| `augmentation_kwargs` | `dict` | `None` | Extra keyword arguments for the augmentation function. |
| `preset` | `str` | `None` | Hyperparameter preset: `"interpretable"`, `"balanced"`, or `"faithful"`. |
| `counterfactual_validity_threshold` | `float` | `None` | If given, filter rules with counterfactual validity score below this threshold (0-1). |
| `counterfactual_noise_scale` | `float` | `0.01` | Perturbation scale for counterfactual boundary probing. |
| `counterfactual_n_probes` | `int` | `20` | Number of random probe pairs per condition for counterfactual scoring. |
| `mdl_selection` | `str` | `None` | MDL selection method: `"forward"`, `"backward"`, or `"score_only"`. |
| `mdl_precision_bits` | `int` or `"auto"` | `"auto"` | Bits per threshold for MDL model cost computation. `"auto"` calibrates from data. |

#### Returns

`ExplanationResult`

#### Raises

- `ValueError` — if both `X_val` and `validation_split` are provided.
- `ValueError` — if `surrogate_type` is not supported.
- `RuntimeError` — if `monotonic_constraints` is provided but sklearn does not support `monotonic_cst`.

---

### `Explainer.cross_validate_fidelity()`

```python
def cross_validate_fidelity(
    self, X, *, y=None, n_folds=5, max_depth=5,
    min_samples_leaf=5, random_state=42,
) -> CVFidelityReport
```

Perform *k*-fold cross-validated fidelity estimation.

---

### `Explainer.compute_stability()`

```python
def compute_stability(
    self, X, *, n_bootstraps=20, max_depth=5,
    min_samples_leaf=5, random_state=42, tolerance=None,
) -> StabilityReport
```

Compute rule stability via bootstrap resampling + Jaccard similarity.

---

### `Explainer.compute_confidence_intervals()`

```python
def compute_confidence_intervals(
    self, result, X, *, y=None, n_bootstraps=1000,
    confidence_level=0.95, random_state=42,
) -> dict[str, ConfidenceInterval]
```

Bootstrap percentile confidence intervals for fidelity (and accuracy if `y` given).

---

### `Explainer.prune_rules()`

```python
def prune_rules(self, result, config) -> ExplanationResult
```

Apply post-hoc pruning to an existing `ExplanationResult`.

---

### `Explainer.extract_stable_rules()`

```python
def extract_stable_rules(
    self, X, *, y=None, n_estimators=20, frequency_threshold=0.5,
    tolerance=0.01, max_depth=5, min_samples_leaf=5, ccp_alpha=0.0,
    monotonic_constraints=None, random_state=42,
    X_val=None, y_val=None,
) -> ExplanationResult
```

Extract stable rules via an ensemble of bootstrap surrogates.

---

### `Explainer.score_rules_counterfactual()`

```python
def score_rules_counterfactual(
    self, result, X, *, validity_threshold=None,
    noise_scale=0.01, n_probes=20, random_state=42,
) -> CounterfactualReport
```

Score extracted rules by counterfactual validity. For each rule condition,
generates multiple random probe pairs straddling the threshold and checks
whether the black-box prediction changes. Using multiple probes makes the
method robust to ensemble models whose boundaries are not axis-aligned.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `ExplanationResult` | *(required)* | A previously computed explanation result. |
| `X` | array-like `(n, p)` | *(required)* | Data used to determine feature ranges. |
| `validity_threshold` | `float` | `None` | If given, rules with score below threshold are filtered. |
| `noise_scale` | `float` | `0.01` | Relative perturbation around each threshold. |
| `n_probes` | `int` | `20` | Number of random probe pairs per condition. A condition is valid if the black-box changes prediction in at least one probe. |
| `random_state` | `int` | `42` | Random seed. |

#### Returns

`CounterfactualReport`

---

### `Explainer.select_rules_mdl()`

```python
def select_rules_mdl(
    self, result, X, *, method="forward", precision_bits="auto",
) -> MDLSelectionReport
```

Select an optimal subset of rules using the MDL principle. Minimises
L(model) + L(data|model) over the extracted rules.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `ExplanationResult` | *(required)* | A previously computed explanation result. |
| `X` | array-like `(n, p)` | *(required)* | Data used to compute coverage and error rates. |
| `method` | `str` | `"forward"` | `"forward"`, `"backward"`, or `"score_only"`. |
| `precision_bits` | `int` or `"auto"` | `"auto"` | Bits per threshold for model cost computation. `"auto"` calibrates from data. |

#### Returns

`MDLSelectionReport`

---

### `Explainer.auto_select_depth()`

```python
def auto_select_depth(
    self, X, *, y=None, target_fidelity=0.85, min_depth=2,
    max_depth=10, n_folds=5, min_samples_leaf=5, random_state=42,
) -> AutoDepthResult
```

Automatically select minimum tree depth achieving target fidelity.

---

### `Explainer.compute_structural_stability()`

```python
def compute_structural_stability(
    self, X, *, n_bootstraps=20, max_depth=5,
    min_samples_leaf=5, top_k=3, random_state=42,
) -> StructuralStabilityReport
```

Compute structural stability: prediction agreement, feature importance
rank stability, and top-k feature agreement across bootstraps.

---

### `Explainer.compute_complementary_metrics()`

```python
def compute_complementary_metrics(self, result, X) -> ComplementaryMetrics
```

Compute metrics beyond standard fidelity: boundary agreement,
counterfactual consistency, effective complexity, per-class fidelity.

---

### `Explainer.compute_fidelity_bounds()`

```python
def compute_fidelity_bounds(self, result, *, delta=0.05) -> FidelityBound
```

Compute theoretical PAC/VC/Rademacher fidelity bounds.

---

### `Explainer.sensitivity_analysis()`

```python
def sensitivity_analysis(
    self, X, *, y=None, depth_range=(3,5,7),
    min_samples_leaf_range=(5,10,20), n_folds=3, random_state=42,
) -> SensitivityResult
```

Grid search over hyperparameters (depth x min_samples_leaf).

---

## `ExplanationResult`

Container returned by `Explainer.extract_rules()`.

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `rules` | `RuleSet` | The extracted IF-THEN rules. |
| `report` | `FidelityReport` | Fidelity metrics. |
| `surrogate` | sklearn tree, `ObliqueTreeSurrogate`, or `SparseObliqueTreeSurrogate` | The fitted surrogate. |
| `train_report` | `FidelityReport \| None` | In-sample report (when hold-out used). |
| `pruned_rules` | `RuleSet \| None` | Pruned rule set. |
| `pruning_report` | `PruningReport \| None` | Summary of pruning operations. |
| `monotonicity_report` | `MonotonicityReport \| None` | Monotonicity validation report. |
| `ensemble_report` | `EnsembleReport \| None` | Ensemble extraction statistics. |
| `stable_rules` | `list[StableRule] \| None` | Stable rules with frequency data. |
| `counterfactual_report` | `CounterfactualReport \| None` | Counterfactual validity report. |
| `mdl_report` | `MDLSelectionReport \| None` | MDL selection report. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `plot(*, save_path=None, **kwargs)` | `None` | Render the surrogate tree via matplotlib. |
| `to_dot()` | `str` | Export the surrogate tree as a Graphviz DOT string. |
| `to_html(output_path)` | `None` | Write an interactive HTML report. |
| `__str__()` | `str` | Human-readable rules + fidelity report. |

---

## `CounterfactualReport`

```python
@dataclass(frozen=True)
class CounterfactualReport
```

Result of counterfactual validity scoring over a full RuleSet.

| Field | Type | Description |
|-------|------|-------------|
| `rule_scores` | `tuple[RuleCounterfactualScore, ...]` | Per-rule scoring breakdown. |
| `filtered_ruleset` | `RuleSet \| None` | Rules above threshold (if threshold given). |
| `validity_threshold` | `float \| None` | The threshold used for filtering. |
| `mean_score` | `float` | Average counterfactual validity score. |
| `std_score` | `float` | Standard deviation of scores. |
| `n_rules_total` | `int` | Rules before filtering. |
| `n_rules_retained` | `int` | Rules after filtering. |
| `noise_scale` | `float` | Perturbation scale used. |
| `random_state` | `int` | Random seed used. |

---

## `RuleCounterfactualScore`

```python
@dataclass(frozen=True)
class RuleCounterfactualScore
```

| Field | Type | Description |
|-------|------|-------------|
| `rule` | `Rule` | The rule being scored. |
| `rule_index` | `int` | Index in the original RuleSet. |
| `score` | `float` | Fraction of conditions that are counterfactually valid (0-1). |
| `condition_validities` | `tuple[ConditionValidity, ...]` | Per-condition breakdown. |
| `n_conditions` | `int` | Total conditions. |
| `n_valid_conditions` | `int` | Valid conditions. |

---

## `ConditionValidity`

```python
@dataclass(frozen=True)
class ConditionValidity
```

| Field | Type | Description |
|-------|------|-------------|
| `condition` | `Condition` | The condition assessed. |
| `is_valid` | `bool` | Whether the black-box changes prediction across this boundary. |
| `bb_changes` | `bool` | Whether the black-box prediction changed. |
| `delta_below` | `float` | Feature value used for the "just below" sample. |
| `delta_above` | `float` | Feature value used for the "just above" sample. |

---

## `MDLSelectionReport`

```python
@dataclass(frozen=True)
class MDLSelectionReport
```

| Field | Type | Description |
|-------|------|-------------|
| `rule_scores` | `tuple[RuleMDLScore, ...]` | MDL decomposition for every rule. |
| `selected_ruleset` | `RuleSet` | The optimal subset selected. |
| `selection_method` | `str` | `"forward"`, `"backward"`, or `"score_only"`. |
| `total_mdl_before` | `float` | Total MDL of the original RuleSet. |
| `total_mdl_after` | `float` | Total MDL of the selected RuleSet. |
| `mdl_reduction` | `float` | Absolute reduction in MDL. |
| `n_rules_original` | `int` | Rules before selection. |
| `n_rules_selected` | `int` | Rules after selection. |
| `precision_bits` | `int` | Bits per threshold used. |

---

## `RuleMDLScore`

```python
@dataclass(frozen=True)
class RuleMDLScore
```

| Field | Type | Description |
|-------|------|-------------|
| `rule` | `Rule` | The rule being scored. |
| `rule_index` | `int` | Index in the RuleSet. |
| `model_cost` | `float` | L(rule) in bits: cost to encode the rule structure. |
| `data_cost` | `float` | L(data\|rule) in bits: cost to encode misclassifications. |
| `total_mdl` | `float` | model_cost + data_cost. |
| `coverage` | `int` | Samples covered by this rule. |
| `error_rate` | `float` | Fraction of covered samples where black-box disagrees. |

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

---

## `RuleSet`

```python
@dataclass(frozen=True)
class RuleSet
```

An ordered collection of rules.

| Field / Property | Type | Description |
|-----------------|------|-------------|
| `rules` | `tuple[Rule, ...]` | Ordered rules. |
| `feature_names` | `tuple[str, ...]` | Feature names. |
| `class_names` | `tuple[str, ...]` | Class names. |
| `num_rules` | `int` | Number of rules. |
| `avg_conditions` | `float` | Mean conditions per rule. |
| `max_conditions` | `int` | Maximum conditions in any rule. |
| `avg_conditions_per_feature` | `float` | `avg_conditions / len(feature_names)`. |
| `interaction_strength` | `float` | Fraction of rules with > 1 distinct feature. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `filter_by_class(class_name)` | `RuleSet` | Subset of rules predicting `class_name`. |
| `rule_signatures()` | `frozenset[str]` | Canonical rule strings. |
| `fuzzy_rule_signatures(tolerance)` | `frozenset[str]` | Fuzzy signatures with rounded thresholds. |
| `to_text()` | `str` | Human-readable multi-line string. |

---

## `FidelityReport`

```python
@dataclass(frozen=True)
class FidelityReport
```

| Field | Type | Description |
|-------|------|-------------|
| `fidelity` | `float` | Agreement between surrogate and black-box (0-1). |
| `accuracy` | `float \| None` | Surrogate accuracy vs true labels. |
| `blackbox_accuracy` | `float \| None` | Black-box accuracy vs true labels. |
| `num_rules` | `int` | Number of rules. |
| `avg_rule_length` | `float` | Mean conditions per rule. |
| `max_rule_length` | `int` | Max conditions per rule. |
| `surrogate_depth` | `int` | Fitted surrogate tree depth. |
| `surrogate_n_leaves` | `int` | Number of leaves. |
| `num_samples` | `int` | Samples used for evaluation. |
| `class_fidelity` | `dict[str, float]` | Per-class fidelity. |
| `evaluation_type` | `str` | `"in_sample"`, `"hold_out"`, `"validation_split"`, or `"cross_validation"`. |
| `avg_conditions_per_feature` | `float \| None` | Normalised complexity. |
| `interaction_strength` | `float \| None` | Multi-feature interaction metric. |
| `fidelity_r2` | `float \| None` | R² surrogate vs black-box (regression). |
| `fidelity_mse` | `float \| None` | MSE surrogate vs black-box (regression). |

---

## `CVFidelityReport`

| Field | Type | Description |
|-------|------|-------------|
| `mean_fidelity` | `float` | Mean fidelity across folds. |
| `std_fidelity` | `float` | Std of fidelity. |
| `mean_accuracy` | `float \| None` | Mean accuracy (if `y` provided). |
| `std_accuracy` | `float \| None` | Std of accuracy. |
| `fold_reports` | `list[FidelityReport]` | Per-fold reports. |
| `n_folds` | `int` | Number of folds. |

---

## `StabilityReport`

| Field | Type | Description |
|-------|------|-------------|
| `mean_jaccard` | `float` | Mean pairwise Jaccard similarity. |
| `std_jaccard` | `float` | Std of pairwise Jaccard. |
| `pairwise_jaccards` | `list[float]` | All C(n,2) pairwise values. |
| `n_bootstraps` | `int` | Number of bootstrap resamples. |

---

## `ConfidenceInterval`

| Field | Type | Description |
|-------|------|-------------|
| `lower` | `float` | Lower bound. |
| `upper` | `float` | Upper bound. |
| `point_estimate` | `float` | Mean of bootstrap distribution. |
| `confidence_level` | `float` | Confidence level (e.g. 0.95). |

---

## `PruningConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_confidence` | `float` | `0.0` | Remove rules below this confidence. |
| `min_samples` | `int` | `0` | Remove rules covering fewer samples. |
| `min_samples_fraction` | `float` | `0.0` | Remove rules covering less than this fraction. |
| `max_conditions` | `int \| None` | `None` | Truncate rules to at most this many conditions. |
| `remove_redundant` | `bool` | `False` | Simplify redundant conditions on the same feature. |

---

## `PruningReport`

| Field | Type | Description |
|-------|------|-------------|
| `original_count` | `int` | Rules before pruning. |
| `pruned_count` | `int` | Rules after pruning. |
| `removed_low_confidence` | `int` | Rules removed by confidence filter. |
| `removed_low_samples` | `int` | Rules removed by sample count filter. |
| `removed_over_max_conditions` | `int` | Rules removed for exceeding max conditions. |
| `conditions_simplified` | `int` | Redundant conditions removed. |

---

## `MonotonicityReport`

| Field | Type | Description |
|-------|------|-------------|
| `constraints` | `dict[str, int]` | The constraint map checked. |
| `violations` | `tuple[MonotonicityViolation, ...]` | All detected violations. |
| `is_compliant` | `bool` | `True` if no violations found. |

---

## `MonotonicityViolation`

| Field | Type | Description |
|-------|------|-------------|
| `rule_index` | `int` | Index of the violating rule. |
| `rule` | `Rule` | The violating rule. |
| `feature` | `str` | Feature with violated constraint. |
| `expected_direction` | `int` | Expected direction (`+1` or `-1`). |
| `description` | `str` | Human-readable description. |

---

## `MonotonicityEnforcementResult`

| Field | Type | Description |
|-------|------|-------------|
| `original_rules` | `RuleSet` | Rules before enforcement. |
| `filtered_rules` | `RuleSet` | Rules after removing violations. |
| `fidelity_before` | `float` | Fidelity before enforcement. |
| `fidelity_after` | `float` | Fidelity after enforcement. |

---

## `StableRule`

| Field | Type | Description |
|-------|------|-------------|
| `rule` | `Rule` | The representative rule instance. |
| `frequency` | `float` | Fraction of trees containing this rule (0-1). |
| `signature` | `str` | Fuzzy signature used for matching. |
| `variant_count` | `int` | Number of slightly different variants merged. |

---

## `EnsembleReport`

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | `int` | Number of bootstrap surrogate trees. |
| `frequency_threshold` | `float` | Minimum frequency to retain a rule. |
| `total_unique_rules` | `int` | Distinct fuzzy signatures across all trees. |
| `stable_rule_count` | `int` | Rules meeting the frequency threshold. |
| `mean_rules_per_tree` | `float` | Average rules per tree. |
| `tolerance` | `float` | Threshold rounding tolerance. |

---

## `HyperparamPreset`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable preset name. |
| `max_depth` | `int` | Tree depth for this preset. |
| `min_samples_leaf` | `int` | Min samples per leaf. |
| `ccp_alpha` | `float` | Cost-complexity pruning alpha. |
| `description` | `str` | Short description of when to use this preset. |

Available presets: `"interpretable"` (3, 20, 0.01), `"balanced"` (5, 10, 0.005), `"faithful"` (8, 5, 0.0).

---

## `AutoDepthResult`

| Field | Type | Description |
|-------|------|-------------|
| `best_depth` | `int` | The shallowest depth achieving target fidelity. |
| `fidelity_scores` | `dict[int, float]` | Mapping from `max_depth` to mean CV fidelity. |
| `selected_fidelity` | `float` | Mean CV fidelity at the selected depth. |
| `target_fidelity` | `float` | Target fidelity threshold used for selection. |

---

## `StructuralStabilityReport`

| Field | Type | Description |
|-------|------|-------------|
| `mean_coverage_overlap` | `float` | Mean coverage overlap across bootstraps. |
| `std_coverage_overlap` | `float` | Std of coverage overlap. |
| `mean_prediction_agreement` | `float` | Mean prediction agreement across bootstrap pairs. |
| `std_prediction_agreement` | `float` | Std of prediction agreement. |
| `feature_importance_stability` | `float` | Feature rank stability via Kendall's tau (0-1). |
| `top_k_feature_agreement` | `float` | Jaccard of top-k feature sets across bootstraps. |
| `n_bootstraps` | `int` | Number of bootstraps. |

---

## `ComplementaryMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `rule_coverage` | `float` | Fraction of samples matched by at least one rule. |
| `boundary_agreement` | `float` | Agreement between surrogate and black-box at decision boundaries. |
| `counterfactual_consistency` | `float` | Consistency of counterfactual changes between surrogate and black-box. |
| `class_balance_fidelity` | `dict[str, float]` | Per-class fidelity. |
| `effective_complexity` | `float` | Fraction of rules that activate on test samples. |

---

## `FidelityBound`

| Field | Type | Description |
|-------|------|-------------|
| `vc_dimension` | `int` | VC dimension of the surrogate tree class. |
| `estimation_error_pac` | `float` | PAC estimation error bound. |
| `estimation_error_rademacher` | `float` | Rademacher estimation error bound. |
| `min_fidelity_pac` | `float` | Lower bound on fidelity (PAC). |
| `min_fidelity_rademacher` | `float` | Lower bound on fidelity (Rademacher). |
| `sample_complexity_required` | `int` | Min samples for epsilon-accuracy. |
| `depth` | `int` | Surrogate tree depth. |
| `n_features` | `int` | Number of features. |
| `n_samples` | `int` | Number of samples. |
| `confidence` | `float` | Confidence level (1 - delta). |
| `max_decision_regions` | `int` | Maximum number of decision regions the tree can represent (2^D). |

---

## `CategoricalMapping`

| Field | Type | Description |
|-------|------|-------------|
| `original_name` | `str` | Original feature name (e.g. `"occupation"`). |
| `encoding` | `str` | `"onehot"` or `"ordinal"`. |
| `encoded_columns` | `tuple[str, ...]` | Names of the encoded columns in the feature matrix. |
| `categories` | `tuple[str, ...]` | Ordered category labels. |

---

## `CategoricalCondition`

Extends `Condition` with a human-readable category value.

| Field | Type | Description |
|-------|------|-------------|
| `feature` | `str` | Original feature name. |
| `operator` | `str` | `"="`, `"in"`, or original (`"<="` / `">"`). |
| `threshold` | `float` | Original numeric threshold. |
| `display_value` | `str` | Decoded category display string (e.g. `"Tech"` or `"{HS, Bachelors, Masters}"`). |

---

## `SensitivityResult`

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[dict]` | List of dicts with keys: `max_depth`, `min_samples_leaf`, `mean_fidelity`, `std_fidelity`, `num_rules`, `avg_rule_length`. |
| `best_config` | `dict` | Configuration dict with the highest fidelity. |
| `best_fidelity` | `float` | Highest mean fidelity found. |

---

## `ObliqueTreeSurrogate`

```python
class ObliqueTreeSurrogate(*, task="classification", max_depth=5,
                           min_samples_leaf=5, max_interaction_features=None,
                           random_state=42)
```

Oblique decision tree surrogate via pairwise feature interaction augmentation.
Adds all pairwise product features to simulate oblique splits.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"classification"` | Task type. |
| `max_depth` | `int \| None` | `5` | Maximum tree depth. |
| `min_samples_leaf` | `int` | `5` | Minimum samples per leaf. |
| `max_interaction_features` | `int \| None` | `None` | Cap on interaction features. `None` = all pairwise. |
| `random_state` | `int` | `42` | Random seed. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X, y)` | `self` | Fit on augmented feature space. |
| `predict(X)` | `ndarray` | Predict using the augmented tree. |
| `get_depth()` | `int` | Fitted tree depth. |
| `get_n_leaves()` | `int` | Number of leaves. |
| `get_augmented_feature_names(feature_names)` | `tuple[str, ...]` | Original + interaction names. |
| `decision_path(X)` | sparse matrix | Decision path in augmented space. |
| `apply(X)` | `ndarray` | Leaf assignments. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tree_` | sklearn tree | Underlying sklearn tree structure. |
| `feature_importances_` | `ndarray` | Feature importances (augmented space). |

---

## `SparseObliqueTreeSurrogate`

```python
class SparseObliqueTreeSurrogate(*, task="classification", max_depth=5,
                                  min_samples_leaf=5, max_iterations=2,
                                  phantom_threshold=0.3, noise_scale=0.01,
                                  n_probes=20, max_interaction_features=None,
                                  random_state=42)
```

Phantom-guided sparse oblique tree. Only adds interaction features for
features involved in phantom splits (splits not supported by the black-box).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"classification"` | Task type. |
| `max_depth` | `int \| None` | `5` | Maximum tree depth. |
| `min_samples_leaf` | `int` | `5` | Minimum samples per leaf. |
| `max_iterations` | `int` | `2` | Maximum phantom-detection iterations. |
| `phantom_threshold` | `float` | `0.3` | Fraction of probes that must NOT change BB for a phantom label. |
| `noise_scale` | `float` | `0.01` | Perturbation magnitude (multiple of feature std). |
| `n_probes` | `int` | `20` | Probe pairs per internal node. |
| `max_interaction_features` | `int \| None` | `None` | Cap on interaction features. Uses MI ranking when set. |
| `random_state` | `int` | `42` | Random seed. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X, y, *, model=None, feature_names=None)` | `self` | Fit with optional phantom detection (requires `model`). |
| `predict(X)` | `ndarray` | Predict using the augmented tree. |
| `get_depth()` | `int` | Fitted tree depth. |
| `get_n_leaves()` | `int` | Number of leaves. |
| `get_augmented_feature_names(feature_names)` | `tuple[str, ...]` | Original + interaction names. |
| `decision_path(X)` | sparse matrix | Decision path in augmented space. |
| `apply(X)` | `ndarray` | Leaf assignments. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tree_` | sklearn tree | Underlying sklearn tree structure. |
| `feature_importances_` | `ndarray` | Feature importances (augmented space). |
| `phantom_features_` | `set[int]` | Original feature indices in phantom splits. |
| `interaction_pairs_` | `list[tuple[int, int]]` | Selected sparse interaction pairs. |
| `n_iterations_` | `int` | Phantom-detection iterations performed. |

---

## Standalone Functions

### `score_rules_counterfactual()`

```python
score_rules_counterfactual(ruleset, model, X, *, validity_threshold=None,
                           noise_scale=0.01, n_probes=20,
                           random_state=42) -> CounterfactualReport
```

Score all rules in a RuleSet by counterfactual validity. Uses multiple
random probe pairs per condition for robustness with ensemble models.

### `select_rules_mdl()`

```python
select_rules_mdl(ruleset, model, X, *, n_classes=2, precision_bits="auto",
                 method="forward") -> MDLSelectionReport
```

Select an optimal subset of rules by minimising total MDL.

### `binary_entropy()`

```python
binary_entropy(p) -> float
```

Compute H(p) = -p*log2(p) - (1-p)*log2(1-p). Returns 0.0 for p in {0, 1}.

### `compute_rule_model_cost()`

```python
compute_rule_model_cost(rule, n_features, n_classes, *, precision_bits=16) -> float
```

Compute L(rule) in bits.

### `compute_fidelity_report()`

```python
compute_fidelity_report(surrogate, X, y_bb, y_true, class_names,
                        num_rules, avg_rule_length, max_rule_length,
                        *, evaluation_type="in_sample",
                        avg_conditions_per_feature=None,
                        interaction_strength=None) -> FidelityReport
```

### `compute_regression_fidelity_report()`

```python
compute_regression_fidelity_report(surrogate, X, y_bb, y_true,
                                   num_rules, avg_rule_length, max_rule_length,
                                   *, evaluation_type="in_sample") -> FidelityReport
```

### `compute_bootstrap_ci()`

```python
compute_bootstrap_ci(values, *, confidence_level=0.95) -> ConfidenceInterval
```

### `augment_data()`

```python
augment_data(X, model, surrogate=None, *, strategy="perturbation",
             n_neighbors=5, noise_scale=0.1, n_boundary_samples=200,
             n_sparse_samples=200, sparsity_quantile=0.25,
             random_state=42) -> tuple[ndarray, ndarray]
```

Augment training data with synthetic samples. Returns `(X_augmented, y_bb_augmented)`.

### `decode_ruleset()`

```python
decode_ruleset(ruleset, mappings) -> RuleSet
```

Decode categorical conditions back to human-readable category names.

### `validate_monotonicity()`

```python
validate_monotonicity(ruleset, constraints) -> MonotonicityReport
```

### `enforce_monotonicity()`

```python
enforce_monotonicity(ruleset, constraints, model, X) -> MonotonicityEnforcementResult
```

Validate and remove violating rules, reporting fidelity impact.

### `filter_monotonic_violations()`

```python
filter_monotonic_violations(ruleset, report) -> RuleSet
```

### `prune_ruleset()`

```python
prune_ruleset(ruleset, config) -> tuple[RuleSet, PruningReport]
```

### `get_preset()`

```python
get_preset(name) -> HyperparamPreset
```

### `auto_select_depth()`

```python
auto_select_depth(explainer, X, *, y=None, target_fidelity=0.85, ...) -> AutoDepthResult
```

### `compute_structural_stability()`

```python
compute_structural_stability(explainer, X, *, n_bootstraps=20, ...) -> StructuralStabilityReport
```

### `compute_complementary_metrics()`

```python
compute_complementary_metrics(surrogate, model, X, ruleset, ...) -> ComplementaryMetrics
```

### `compute_fidelity_bounds()`

```python
compute_fidelity_bounds(depth, n_features, n_samples,
                        empirical_infidelity=0.0, delta=0.05) -> FidelityBound
```

Compute all theoretical fidelity bounds. `empirical_infidelity` is `1 - fidelity`;
`delta` is the failure probability (bounds hold with prob >= 1 - delta).

### `compute_adaptive_tolerance()`

```python
compute_adaptive_tolerance(X, feature_names, *, scale=0.05) -> dict[str, float]
```

Compute per-feature fuzzy matching tolerances as a fraction of each feature's
standard deviation. Returns a dict mapping feature name to tolerance value.

### Theoretical Bound Utilities

```python
vc_dimension_decision_tree(depth, n_features) -> int
estimation_error_pac(vc_dim, n_samples, delta) -> float
estimation_error_rademacher(depth, n_features, n_samples, delta) -> float
sample_complexity(vc_dim, epsilon, delta) -> int
relu_network_regions(n_layers, width, input_dim) -> int
min_depth_for_regions(n_regions) -> int
optimal_depth_bound(n_samples, n_features) -> float
```

### Visualization

```python
plot_surrogate_tree(surrogate, feature_names, class_names=(), ...) -> None
export_dot(surrogate, feature_names, class_names=()) -> str
export_html(result, output_path) -> None
```
