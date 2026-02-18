# Methodology

This document describes the scientific foundations of TRACE, including the
surrogate approximation approach, fidelity metrics, statistical validation
procedures, and complexity measures.

---

## 1. Surrogate Approximation

### 1.1 Problem Setting

Let *f* : X → Y be a black-box classifier. We want a
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

The prediction is the majority class at the leaf; the **confidence** is the
fraction of the dominant class.

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

### 2.2 Accuracy (vs. True Labels)

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

## 6. Regulatory Compliance Features

### 6.1 Post-Hoc Rule Pruning

Deep surrogates produce rules with many conditions that are difficult to
justify to auditors. TRACE provides configurable post-hoc pruning:

**Confidence filtering:** Rules with confidence below a threshold (e.g. 0.6)
are removed. Low-confidence rules indicate leaves where the surrogate is
uncertain and provide weak explanatory value.

**Sample coverage filtering:** Rules covering fewer than *k* samples (or less
than a fraction of the dataset) are removed. Rare rules are statistically
unreliable and prone to overfitting.

**Redundant condition simplification:** For conditions on the same feature,
redundant splits are collapsed:
- `A > 5 AND A > 3` → `A > 5` (keep tightest lower bound)
- `A ≤ 10 AND A ≤ 7` → `A ≤ 7` (keep tightest upper bound)

**Condition truncation:** Rules with more than *N* conditions (typically 3–4
for regulatory contexts) are truncated to the first *N* conditions. Conditions
closer to the tree root are more discriminative and kept preferentially.

**Cost-complexity pruning:** The `ccp_alpha` parameter is passed directly to
sklearn's decision tree, which performs minimal cost-complexity pruning during
fitting (Breiman et al., 1984). This is complementary to the post-hoc filters.

### 6.2 Monotonicity Constraints

Financial regulations often require that certain inputs have a univocal effect
on the output. For example, increasing income should not increase credit risk.

TRACE supports two layers of monotonicity enforcement:

**During fitting:** When `monotonic_constraints` is provided, the `monotonic_cst`
parameter is passed to sklearn's `DecisionTreeClassifier`/`DecisionTreeRegressor`,
which enforces the constraints during tree construction. This guarantees that
splits on constrained features respect the declared direction.

**Post-hoc validation:** After rule extraction, `validate_monotonicity()` checks
all rule pairs for violations. For regression, it verifies that prediction
values are monotonically ordered with respect to constrained features. For
classification, it flags cases where the predicted class changes inconsistently
as a constrained feature increases. A `MonotonicityReport` with any detected
violations is attached to the result.

### 6.3 Ensemble Rule Extraction (Bagging)

A single surrogate tree is sensitive to the training data: small perturbations
can produce different rules. For regulatory audits, rule stability is
essential — re-running the process tomorrow should not produce drastically
different rules.

TRACE addresses this via ensemble rule extraction:

1. Generate *N* bootstrap samples of the explanation data.
2. Train a separate surrogate tree on each sample.
3. Extract rules from each tree and compute **fuzzy signatures** (thresholds
   rounded to a configurable tolerance) so that near-identical rules are
   recognized as the same rule.
4. Count how many trees each rule appears in.
5. Retain only rules appearing in at least a specified fraction (e.g. 50%)
   of the trees.

The fuzzy matching addresses a limitation of the existing Jaccard stability
metric, where threshold differences of 0.0001 caused rules to be treated as
entirely different.

The `EnsembleReport` provides statistics on the extraction process, including
the number of unique rules, stable rule count, and mean rules per tree.

---

## 7. Data Augmentation

### 7.1 Motivation

Standard surrogate training uses only the available data *X* without exploring
under-represented regions. This can lead to surrogates that are faithful only
where data is dense, missing important decision boundaries in sparse areas.

### 7.2 Augmentation Strategies

TRACE implements three complementary strategies:

**Perturbation augmentation (LIME-style):** For each sample in *X*, generate
*k* neighbours by adding Gaussian noise scaled by feature-wise standard
deviation. The black-box is queried on each synthetic sample to produce
pseudo-labels. This densifies the neighbourhood of existing points.

**Boundary augmentation:** Samples are generated near detected decision
boundaries by interpolating between pairs of samples with different
black-box predictions. This focuses the surrogate's training on the most
informative regions.

**Sparse region augmentation:** Samples are generated uniformly in
under-represented regions of the feature space (identified by low local
density). This addresses coverage gaps that could lead to unreliable rules.

All three strategies can be combined via `augment_data(strategy="combined")`.

---

## 8. Oblique Surrogate Trees

### 8.1 Motivation

Standard axis-aligned decision trees split on one feature at a time. When the
black-box model has diagonal decision boundaries (e.g. `x₁ + x₂ > 5`), the
surrogate requires additional depth to approximate them via a staircase of
axis-aligned splits. This increases rule complexity and may introduce phantom
splits — boundaries that exist in the surrogate but not in the black-box.

### 8.2 Oblique Tree Surrogate

The `ObliqueTreeSurrogate` approximates oblique splits by augmenting the feature
space with pairwise interaction products. For *p* original features, it adds
up to *p(p-1)/2* new features of the form `feature_i × feature_j`. A standard
axis-aligned tree fitted on the augmented space can then split on conditions
like `feature_A × feature_B ≤ θ`, which corresponds to an oblique hyperplane
in the original space.

**Trade-off:** The augmented feature space grows quadratically with the number
of features. For high-dimensional data, the `max_interaction_features` parameter
caps the number of interaction features.

### 8.3 Sparse Oblique Tree Surrogate

The `SparseObliqueTreeSurrogate` addresses the quadratic growth problem by
adding interaction features only where needed. It uses a phantom-guided
iterative algorithm:

1. **Fit tree₀:** Train a standard axis-aligned tree on (X, ŷ).
2. **Detect phantoms:** For each internal node, generate random probe pairs
   straddling the split threshold. If the black-box prediction does not change
   for any probe (below `phantom_threshold`), the split is a phantom.
3. **Add sparse interactions:** Create product features only for features
   involved in phantom splits. When `max_interaction_features` is set, rank
   pairs by mutual information with the target and keep the top-k.
4. **Re-fit:** Train a new tree on the augmented space.
5. **Iterate:** Repeat steps 2–4 up to `max_iterations` or until no new
   phantom features are detected (convergence).

When no black-box model is available (fallback mode), interaction pairs are
selected from the top-k most important features in tree₀.

This approach typically adds far fewer interaction features than the full
oblique tree while resolving the phantom splits that matter most.

---

## 9. Theoretical Fidelity Bounds

### 9.1 VC-Dimension of Decision Trees

A binary decision tree of depth *D* over *p* features has VC dimension
bounded by:

```
VC(D, p) ≈ 2^D - 1
```

This quantifies the expressiveness of the surrogate hypothesis class and
determines the sample complexity required for PAC-learning guarantees.

### 9.2 PAC-Learning Bound

Using the agnostic PAC-learning framework (Blumer et al., 1989), the
estimation error is bounded by:

```
ε_pac ≤ √( (VC · ln(2N/VC) + ln(4/δ)) / (2N) )
```

where *N* is the number of samples and *δ* is the failure probability. This
gives a worst-case guarantee on how much the empirical fidelity can deviate
from the true fidelity.

### 9.3 Rademacher Complexity Bound

A tighter, data-dependent bound based on Rademacher complexity
(Bartlett & Mendelson, 2002):

```
ε_rad ≤ 2 · √(VC · ln(eN/VC) / N) + √(ln(1/δ) / (2N))
```

### 9.4 Sample Complexity

Inverting the PAC bound gives the minimum number of samples required to
guarantee estimation error ≤ ε with confidence 1 − δ:

```
N ≥ (VC · ln(2/ε) + ln(4/δ)) / (2ε²)
```

### 9.5 Neural Network Approximation

For ReLU networks, TRACE can estimate the number of linear regions
(Montufar et al., 2014) and determine the minimum tree depth required
to partition all regions:

```
R(network) ≤ ∏_l min(n_l choose d, 1)^l · 2^d
D_min ≥ ⌈log₂(R)⌉
```

---

## 10. Counterfactual Rule Scoring

### 10.1 Motivation

A high-fidelity surrogate may still place individual splits at locations that
do not reflect meaningful black-box decision boundaries. For example, a
surrogate might split at `income ≤ 50000` with 98% fidelity, but the black-box
might not actually change its prediction around that threshold — the split is
a "phantom boundary" introduced by the surrogate's learning procedure.

### 10.2 Algorithm

For each condition in a rule (e.g. `feature_j ≤ θ`):

1. Generate *n_probes* random base samples within the observed feature ranges.
2. For each probe, create two copies: one with `feature_j = θ − δ` (below)
   and one with `feature_j = θ + δ` (above), where `δ = noise_scale × σ_j`.
3. Query the black-box on both samples.
4. If the prediction changes in **at least one probe**, the condition is
   **counterfactually valid** — the black-box also treats this threshold
   as a boundary.

Using multiple probes is essential for ensemble models (GBM, Random Forest)
whose decision boundaries are not axis-aligned.  A single probe pair may
land in a region where other features dominate the prediction, causing the
boundary to appear non-existent.  With *n_probes* random base points, the
probability of missing a real boundary decreases exponentially.

The rule's **validity score** is the fraction of valid conditions (0.0–1.0).
Rules with low scores contain "phantom splits" that the black-box ignores.

### 10.3 Filtering

When a `validity_threshold` is specified, rules scoring below the threshold
are removed. This produces explanations where every remaining rule boundary
corresponds to a real black-box decision boundary, addressing the fundamental
limitation of surrogate-based explainers.

---

## 11. MDL-Based Rule Selection

### 11.1 The Minimum Description Length Principle

The MDL principle (Rissanen, 1978; Grunwald, 2007) selects models that best
compress the data. For rule selection:

```
MDL(rule) = L(model) + L(data | model)
```

- **L(model)** — bits to encode the rule structure:
  `n_conditions × (log₂(n_features) + precision_bits) + log₂(n_classes)`
- **L(data | model)** — bits to encode misclassifications on covered samples:
  `coverage × H(error_rate)`, where H is the binary entropy function.

### 11.2 Selection Algorithms

**Forward selection:** Sort rules by ascending total MDL. Include a rule only
if its total MDL is less than the null-hypothesis cost (`log₂(n_classes) ×
coverage`). This produces a minimal ruleset.

**Backward elimination:** Start with all rules. Iteratively remove the rule
whose cost exceeds its null savings by the largest margin. Stop when no rule
can be profitably removed.

### 11.3 Complementarity with Other Methods

MDL selection is complementary to both frequency-based ensemble selection and
counterfactual scoring:

- **Ensemble selection** identifies rules that are **stable** across bootstraps.
- **Counterfactual scoring** identifies rules whose boundaries are **valid**.
- **MDL selection** identifies rules that are **information-efficient**.

The three can be chained: ensemble → counterfactual → MDL.

---

## 12. Structural Stability

### 12.1 Beyond Jaccard Similarity

Syntactic Jaccard similarity compares rule strings exactly. Two rulesets
covering identical regions but with threshold differences of 0.0001 score
Jaccard = 0. TRACE provides three semantic stability metrics:

**Coverage overlap:** Measures whether bootstrap rulesets cover the same
samples, regardless of how the rules are expressed.

**Prediction agreement:** Measures whether bootstrap surrogates make the same
predictions on the evaluation data.

**Feature importance stability:** Measures whether bootstrap surrogates
rank features similarly (cosine similarity of feature importance vectors).

---

## 13. Theoretical Considerations

### 13.1 Fidelity–Interpretability Trade-off

Increasing `max_depth` improves fidelity at the cost of more rules. In the
limit (`max_depth = None`), the surrogate perfectly memorises the black-box on
the training data, but the rules become too numerous to interpret.

Practitioners should select the shallowest tree that achieves acceptable
fidelity. The cross-validated fidelity and stability analyses help determine
this threshold.

### 13.2 Limitations

- **Global approximation**: the surrogate is a single global model. It may miss
  local decision patterns that are important for specific instances. For local
  explanations, consider LIME or SHAP as complements.

- **Axis-aligned splits**: standard decision trees split on one feature at a
  time. If the black-box uses oblique boundaries (e.g. `x₁ + x₂ > 5`), the
  surrogate may need more depth to approximate them. The `ObliqueTreeSurrogate`
  and `SparseObliqueTreeSurrogate` mitigate this by adding interaction features
  (see Section 8).

- **Discrete approximation**: the surrogate uses step functions (piece-wise
  constant predictions in each leaf), which may poorly approximate smooth
  regression surfaces.

- **Training data dependency**: the quality of the surrogate depends on the
  quality and coverage of the explanation data *X*. If *X* does not cover the
  full input space, the rules are valid only within the observed data
  distribution.

---

## 14. References

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

8. Rissanen, J. (1978). *Modeling by shortest data description*. Automatica,
   14(5), 465-471.

9. Grunwald, P. (2007). *The Minimum Description Length Principle*. MIT Press.

10. Blumer, A., Ehrenfeucht, A., Haussler, D., & Warmuth, M. (1989).
    *Learnability and the Vapnik-Chervonenkis Dimension*. Journal of the ACM.

11. Bartlett, P. L., & Mendelson, S. (2002). *Rademacher and Gaussian
    Complexities: Risk Bounds and Structural Results*. JMLR.

12. Montufar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). *On the Number
    of Linear Regions of Deep Neural Networks*. NeurIPS.
