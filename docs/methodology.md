# Methodology

This document describes the scientific foundations of TRACE, including the
surrogate approximation approach, fidelity metrics, statistical validation
procedures, and complexity measures.

---

## 1. Surrogate Approximation

### 1.1 Problem Setting

Let *f* : X → Y be a black-box model (classifier or regressor). We want a
**global, interpretable approximation** *g* such that:

1. *g(x) ≈ f(x)* for all x in the data distribution (high **fidelity**).
2. *g* is human-interpretable (low **complexity**).

TRACE uses a shallow **decision tree** as the surrogate *g*. Decision trees
produce IF-THEN rules that are directly readable, and their depth controls
the interpretability–fidelity trade-off.

### 1.2 Training Procedure

Given training data *X* and the black-box model *f*:

1. Compute black-box predictions: *ŷ = f(X)*.
2. Fit a decision tree *g* on *(X, ŷ)* with hyperparameters `max_depth` and
   `min_samples_leaf`.
3. Extract one IF-THEN rule per leaf via depth-first traversal.

The surrogate is trained to **mimic the black-box**, not to predict the true
labels. This is intentional: the rules describe *what the model learned*, not
what the ground truth is.

### 1.3 Rule Extraction

Each leaf of the decision tree corresponds to a rule:

```
IF condition_1 AND condition_2 AND ... AND condition_k THEN prediction
```

Where each condition is a threshold split on a feature: `feature_j ≤ θ` or
`feature_j > θ`. The path from the root to a leaf defines the conjunction of
conditions.

For classification, the prediction is the majority class at the leaf; the
**confidence** is the fraction of the dominant class. For regression, the
prediction is the mean target value.

---

## 2. Fidelity Metrics

### 2.1 Classification Fidelity

**Fidelity** is the agreement rate between the surrogate and the black-box:

```
Fidelity = (1/n) Σᵢ 𝟙[g(xᵢ) = f(xᵢ)]
```

This is equivalent to the accuracy of the surrogate when treating the black-box
predictions as ground truth.

**Per-class fidelity** is computed for each class *c* by restricting the
evaluation to samples where *f(xᵢ) = c*:

```
Fidelity_c = (1/nₒ) Σ_{i: f(xᵢ)=c} 𝟙[g(xᵢ) = c]
```

This reveals whether the surrogate is better at mimicking certain classes than
others.

### 2.2 Regression Fidelity

For regression, fidelity is measured by the **coefficient of determination**
(R²) between surrogate predictions and black-box predictions:

```
Fidelity_R² = 1 - Σᵢ(g(xᵢ) - f(xᵢ))² / Σᵢ(f(xᵢ) - f̄)²
```

The **mean squared error** (MSE) is also reported:

```
Fidelity_MSE = (1/n) Σᵢ(g(xᵢ) - f(xᵢ))²
```

### 2.3 Accuracy (vs. True Labels)

When true labels *y* are provided, TRACE also reports:

- **Surrogate accuracy**: how well the surrogate predicts the true labels.
- **Black-box accuracy**: how well the original model predicts the true labels.

This allows assessing whether the interpretable surrogate sacrifices real-world
performance.

---

## 3. Evaluation Protocols

### 3.1 In-sample Evaluation (Default)

Fidelity is computed on the same data used to train the surrogate. This gives
an **upper bound** on true fidelity and may be optimistically biased,
especially with deep surrogates.

### 3.2 Hold-out Evaluation

The user provides a separate validation set (`X_val`) or requests an internal
split (`validation_split`). The surrogate is trained on the training portion
and evaluated on the held-out portion:

- `evaluation_type = "hold_out"` (external validation set)
- `evaluation_type = "validation_split"` (automatic internal split)

For classification with `validation_split`, the split is **stratified** to
preserve class proportions.

### 3.3 Cross-Validated Fidelity

*k*-fold cross-validation provides a robust fidelity estimate with reduced
variance:

1. Split *X* into *k* folds (stratified for classification).
2. For each fold, train the surrogate on *k-1* folds and evaluate on the
   held-out fold.
3. Report mean ± std fidelity across folds.

This method re-trains a **fresh surrogate** for each fold, ensuring no data
leakage between training and evaluation.

---

## 4. Statistical Validation

### 4.1 Bootstrap Confidence Intervals

Given a fitted surrogate and evaluation data, the bootstrap CI procedure:

1. For *B* iterations (default 1000):
   - Draw a bootstrap sample of indices with replacement.
   - Compute fidelity (and accuracy) on the bootstrap sample.
2. Report the percentile interval [α/2, 1-α/2] of the *B* bootstrap values.

This is the **percentile bootstrap** method (Efron & Tibshirani, 1993). It
provides a non-parametric confidence interval that accounts for the variability
in the evaluation data.

**Note:** the confidence intervals quantify uncertainty in the **evaluation
metric**, not in the surrogate model itself. The surrogate is fixed; the CI
reflects sampling variability in the test set.

### 4.2 Bootstrap Stability (Jaccard)

Rule stability measures how sensitive the extracted rules are to perturbations
in the training data:

1. Generate *B* bootstrap resamples of *X* (default *B* = 20).
2. For each resample, train a fresh surrogate and extract rules.
3. Represent each rule set as a **canonical signature set** (sorted conditions
   + prediction).
4. Compute all C(*B*, 2) pairwise Jaccard similarities:

```
J(Aᵢ, Aⱼ) = |Aᵢ ∩ Aⱼ| / |Aᵢ ∪ Aⱼ|
```

5. Report mean ± std Jaccard.

**Interpretation:**

| Mean Jaccard | Assessment |
|-------------|------------|
| 0.8 – 1.0 | Highly stable: same rules emerge regardless of training variation. |
| 0.5 – 0.8 | Moderately stable: core rules persist, but some variation. |
| 0.0 – 0.5 | Unstable: rules are sensitive to data perturbation. Consider simpler surrogates. |

---

## 5. Complexity Metrics

### 5.1 Standard Metrics

| Metric | Definition |
|--------|-----------|
| `num_rules` | Number of leaves in the surrogate tree. |
| `avg_conditions` | Mean number of conditions per rule (= mean path length). |
| `max_conditions` | Maximum conditions in any single rule. |
| `surrogate_depth` | Depth of the fitted tree. |

### 5.2 Normalised Metrics

Standard metrics are not comparable across datasets with different
dimensionalities. TRACE introduces two normalised metrics:

**`avg_conditions_per_feature`:**

```
avg_conditions_per_feature = avg_conditions / p
```

where *p* is the number of features. A ruleset with average 3 conditions over
100 features (0.03) is much simpler, relatively, than one with average 3
conditions over 4 features (0.75).

**`interaction_strength`:**

```
interaction_strength = (1/R) Σᵣ 𝟙[|features(ruleᵣ)| > 1]
```

where *R* is the total number of rules and `features(ruleᵣ)` is the set of
distinct features referenced in rule *r*. Values near 0 indicate univariate
(axis-aligned) surrogates; values near 1 indicate heavy multi-feature
interactions.

---

## 6. Theoretical Considerations

### 6.1 Fidelity–Interpretability Trade-off

Increasing `max_depth` improves fidelity at the cost of more rules. In the
limit (`max_depth = None`), the surrogate perfectly memorises the black-box on
the training data, but the rules become too numerous to interpret.

Practitioners should select the shallowest tree that achieves acceptable
fidelity. The cross-validated fidelity and stability analyses help determine
this threshold.

### 6.2 Limitations

- **Global approximation**: the surrogate is a single global model. It may miss
  local decision patterns that are important for specific instances. For local
  explanations, consider LIME or SHAP as complements.

- **Axis-aligned splits**: decision trees split on one feature at a time. If
  the black-box uses oblique boundaries (e.g. `x₁ + x₂ > 5`), the surrogate
  may need more depth to approximate them.

- **Discrete approximation**: the surrogate uses step functions (piece-wise
  constant predictions in each leaf), which may poorly approximate smooth
  regression surfaces.

- **Training data dependency**: the quality of the surrogate depends on the
  quality and coverage of the explanation data *X*. If *X* does not cover the
  full input space, the rules are valid only within the observed data
  distribution.

---

## 7. References

1. Craven, M. W., & Shavlik, J. W. (1996). *Extracting tree-structured
   representations of trained networks*. Advances in Neural Information
   Processing Systems.

2. Bastani, O., Kim, C., & Bastani, H. (2017). *Interpreting Blackbox Models
   via Model Extraction*. arXiv:1705.08504.

3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust
   You?": Explaining the Predictions of Any Classifier*. KDD 2016 (LIME).

4. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting
   Model Predictions*. NeurIPS 2017 (SHAP).

5. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
   Chapman & Hall/CRC.

6. Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., &
   Pedreschi, D. (2018). *A Survey of Methods for Explaining Black Box
   Models*. ACM Computing Surveys.

7. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making
   Black Box Models Explainable*. 2nd ed. christophm.github.io/
   interpretable-ml-book.
