#!/usr/bin/env python3
"""
Complete demo: every trace_xai feature in a single script.

Demonstrates all public functions on the Iris dataset (classification).
Each section is self-contained and prints labelled output so you can
run it end-to-end.

Usage:
    python examples/demo_all_features.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import trace_xai
from trace_xai import (
    # Core
    Explainer,
    # Pruning
    PruningConfig,
    # Monotonicity
    validate_monotonicity,
    filter_monotonic_violations,
    enforce_monotonicity,
    # Augmentation
    augment_data,
    perturbation_augmentation,
    boundary_augmentation,
    sparse_region_augmentation,
    # Categorical decoding
    CategoricalMapping,
    decode_ruleset,
    decode_conditions,
    # Hyperparameter presets
    get_preset,
    auto_select_depth,
    sensitivity_analysis,
    compute_adaptive_tolerance,
    PRESETS,
    # Structural stability
    compute_structural_stability,
    # Complementary metrics
    compute_complementary_metrics,
    # Theoretical bounds
    compute_fidelity_bounds,
    vc_dimension_decision_tree,
    estimation_error_pac,
    estimation_error_rademacher,
    sample_complexity,
    relu_network_regions,
    min_depth_for_regions,
    optimal_depth_bound,
    # Counterfactual scoring
    score_rules_counterfactual,
    # MDL selection
    select_rules_mdl,
    binary_entropy,
    compute_rule_model_cost,
    # Surrogates
    SparseObliqueTreeSurrogate,
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Setup: Iris classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("Setup: Iris classification dataset")

iris = load_iris()
X, y = iris.data, iris.target
feature_names = list(iris.feature_names)
class_names = list(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

explainer = Explainer(
    clf,
    feature_names=feature_names,
    class_names=class_names,
)

print(f"Samples: train={len(X_train)}, test={len(X_test)}")
print(f"Features: {feature_names}")
print(f"Classes: {class_names}")
print(f"Black-box test accuracy: {clf.score(X_test, y_test):.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Basic rule extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("1. Basic rule extraction (extract_rules)")

result = explainer.extract_rules(X_train, y=y_train, max_depth=4)

print(result.rules)
print()
print(result.report)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Hold-out fidelity evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("2. Hold-out fidelity evaluation")

result_holdout = explainer.extract_rules(
    X_train, y=y_train,
    X_val=X_test, y_val=y_test,
    max_depth=4,
)

print(f"Hold-out report (evaluation_type={result_holdout.report.evaluation_type}):")
print(result_holdout.report)
print(f"\nTrain report (evaluation_type={result_holdout.train_report.evaluation_type}):")
print(result_holdout.train_report)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Validation split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("3. Automatic validation split")

result_split = explainer.extract_rules(
    X_train, y=y_train,
    validation_split=0.2,
    max_depth=4,
)

print(f"Evaluation type: {result_split.report.evaluation_type}")
print(f"Fidelity: {result_split.report.fidelity:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Cross-validated fidelity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("4. Cross-validated fidelity (cross_validate_fidelity)")

cv_report = explainer.cross_validate_fidelity(
    X_train, y=y_train,
    n_folds=5,
    max_depth=4,
    random_state=42,
)

print(cv_report)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Bootstrap stability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("5. Bootstrap stability (compute_stability)")

stability = explainer.compute_stability(
    X_train,
    n_bootstraps=10,
    max_depth=4,
    random_state=42,
)

print(stability)

# Also try fuzzy matching
fuzzy_stability = explainer.compute_stability(
    X_train,
    n_bootstraps=10,
    max_depth=4,
    tolerance=0.1,
    random_state=42,
)

print(f"\nExact Jaccard: {stability.mean_jaccard:.4f}")
print(f"Fuzzy Jaccard: {fuzzy_stability.mean_jaccard:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Bootstrap confidence intervals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("6. Bootstrap confidence intervals (compute_confidence_intervals)")

cis = explainer.compute_confidence_intervals(
    result, X_test,
    y=y_test,
    n_bootstraps=500,
    confidence_level=0.95,
    random_state=42,
)

fid_ci = cis["fidelity"]
print(f"Fidelity: {fid_ci.point_estimate:.4f} "
      f"[{fid_ci.lower:.4f}, {fid_ci.upper:.4f}] (95% CI)")

if "accuracy" in cis:
    acc_ci = cis["accuracy"]
    print(f"Accuracy: {acc_ci.point_estimate:.4f} "
          f"[{acc_ci.lower:.4f}, {acc_ci.upper:.4f}] (95% CI)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Normalised complexity metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("7. Normalised complexity metrics")

print(f"Avg conditions per feature: {result.report.avg_conditions_per_feature:.4f}")
print(f"Interaction strength:       {result.report.interaction_strength:.4f}")
print(f"Num rules:                  {result.report.num_rules}")
print(f"Avg rule length:            {result.report.avg_rule_length:.2f}")
print(f"Max rule length:            {result.report.max_rule_length}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Regulatory pruning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("8. Regulatory pruning (PruningConfig)")

# 8a. Inline pruning during extraction
result_pruned = explainer.extract_rules(
    X_train, y=y_train,
    max_depth=5,
    pruning=PruningConfig(
        min_confidence=0.6,
        min_samples=5,
        max_conditions=4,
        remove_redundant=True,
    ),
)

print(f"Original rules: {result_pruned.rules.num_rules}")
print(f"Pruned rules:   {result_pruned.pruned_rules.num_rules}")
print(f"\nPruning report:\n{result_pruned.pruning_report}")

# 8b. Post-hoc pruning
pruned_result = explainer.prune_rules(
    result,
    PruningConfig(min_confidence=0.8, max_conditions=3),
)
print(f"\nPost-hoc pruning: {result.rules.num_rules} -> {pruned_result.pruned_rules.num_rules} rules")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Monotonicity constraints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("9. Monotonicity constraints")

# Note: sklearn monotonic_cst does not support multiclass.
# We demonstrate on the regression dataset later, and here we show
# post-hoc validation/filtering which DOES work on multiclass.

# 9a. Post-hoc validation
mono_report = validate_monotonicity(
    result.rules,
    {"petal length (cm)": 1, "petal width (cm)": 1},
)
print(f"Post-hoc validation: compliant={mono_report.is_compliant}, "
      f"violations={len(mono_report.violations)}")

if not mono_report.is_compliant:
    for v in mono_report.violations[:3]:
        print(f"  Violation: {v.description}")

# 9b. Filter violating rules
clean_rules = filter_monotonic_violations(result.rules, mono_report)
print(f"\nRules after filtering: {clean_rules.num_rules} "
      f"(was {result.rules.num_rules})")

# 9c. Enforce monotonicity
enforcement = enforce_monotonicity(
    result.rules,
    {"petal length (cm)": 1, "petal width (cm)": 1},
    surrogate=result.surrogate,
    X=X_train,
    model=clf,
)
print(f"Enforcement: rules_removed={enforcement.rules_removed}, "
      f"fidelity_impact={enforcement.fidelity_impact}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Ensemble rule extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("10. Ensemble rule extraction (extract_stable_rules)")

result_ensemble = explainer.extract_stable_rules(
    X_train, y=y_train,
    n_estimators=15,
    frequency_threshold=0.3,
    tolerance=0.1,
    max_depth=4,
    random_state=42,
)

print(result_ensemble.ensemble_report)
print(f"\nStable rules: {len(result_ensemble.stable_rules)}")
for sr in result_ensemble.stable_rules[:5]:
    print(f"  [{sr.frequency:.0%}] {sr.rule}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Data augmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("11. Data augmentation")

# 11a. Perturbation augmentation
X_pert, y_pert = perturbation_augmentation(
    X_train, clf,
    n_neighbors=3,
    noise_scale=0.1,
    random_state=42,
)
print(f"Perturbation augmentation: {X_train.shape[0]} -> {X_pert.shape[0]} samples")

# 11b. Boundary augmentation (needs a surrogate)
X_bound, y_bound = boundary_augmentation(
    X_train, clf, result.surrogate,
    n_samples=100,
    random_state=42,
)
print(f"Boundary augmentation: +{X_bound.shape[0]} boundary samples")

# 11c. Sparse region augmentation
X_sparse, y_sparse = sparse_region_augmentation(
    X_train, clf, result.surrogate,
    n_samples=100,
    random_state=42,
)
print(f"Sparse region augmentation: +{X_sparse.shape[0]} sparse samples")

# 11d. Combined augmentation
X_aug, y_aug = augment_data(
    X_train, clf, result.surrogate,
    strategy="combined",
    n_neighbors=3,
    n_boundary_samples=100,
    n_sparse_samples=100,
    random_state=42,
)
print(f"Combined augmentation: {X_train.shape[0]} -> {X_aug.shape[0]} samples")

# 11e. Use augmented data for extraction
result_aug = explainer.extract_rules(X_aug, max_depth=4)
print(f"Fidelity on augmented data: {result_aug.report.fidelity:.4f}")

# 11f. Augmentation via extract_rules parameter
result_inline_aug = explainer.extract_rules(
    X_train, y=y_train,
    max_depth=4,
    augmentation="perturbation",
    augmentation_kwargs={"n_neighbors": 3, "noise_scale": 0.1},
)
print(f"Fidelity with inline augmentation: {result_inline_aug.report.fidelity:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Categorical feature decoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("12. Categorical feature decoding")

# Simulate a CategoricalMapping (Iris has no categoricals, so we demo the API)
mappings = [
    CategoricalMapping(
        original_name="species_group",
        encoding="onehot",
        encoded_columns=("species_group_A", "species_group_B"),
        categories=("GroupA", "GroupB"),
    ),
]

print(f"CategoricalMapping: {mappings[0]}")
print(f"  original_name: {mappings[0].original_name}")
print(f"  encoding:      {mappings[0].encoding}")
print(f"  categories:    {mappings[0].categories}")

# decode_ruleset and decode_conditions work when rule conditions reference
# the encoded column names.  With Iris features they pass through unchanged.
decoded = decode_ruleset(result.rules, mappings)
print(f"\ndecode_ruleset: {result.rules.num_rules} rules (pass-through, no matching columns)")

# Individual condition decoding
if result.rules.rules:
    decoded_conds = decode_conditions(result.rules.rules[0].conditions, mappings)
    print(f"decode_conditions on first rule: {len(decoded_conds)} conditions")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Hyperparameter presets & auto-depth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("13. Hyperparameter presets & auto-depth")

# 13a. List all presets
print("Available presets:")
for name, preset in PRESETS.items():
    print(f"  {name}: max_depth={preset.max_depth}, "
          f"min_samples_leaf={preset.min_samples_leaf}, "
          f"ccp_alpha={preset.ccp_alpha}")

# 13b. Get a specific preset
balanced = get_preset("balanced")
print(f"\nget_preset('balanced'): {balanced}")

# 13c. Use preset in extraction
result_preset = explainer.extract_rules(X_train, y=y_train, preset="interpretable")
print(f"\nPreset 'interpretable': {result_preset.rules.num_rules} rules, "
      f"fidelity={result_preset.report.fidelity:.4f}")

result_faithful = explainer.extract_rules(X_train, y=y_train, preset="faithful")
print(f"Preset 'faithful':      {result_faithful.rules.num_rules} rules, "
      f"fidelity={result_faithful.report.fidelity:.4f}")

# 13d. Auto-select depth
depth_result = auto_select_depth(
    explainer, X_train,
    y=y_train,
    min_depth=2,
    max_depth=8,
    target_fidelity=0.95,
    n_folds=3,
    random_state=42,
)
print(f"\nauto_select_depth: best_depth={depth_result.best_depth}, "
      f"fidelity={depth_result.selected_fidelity:.4f}")
print(f"  Fidelity per depth: {depth_result.fidelity_scores}")

# 13e. Also via the Explainer method
depth_result2 = explainer.auto_select_depth(
    X_train, y=y_train,
    target_fidelity=0.95,
    min_depth=2,
    max_depth=8,
    random_state=42,
)
print(f"  (via Explainer) best_depth={depth_result2.best_depth}")

# 13f. Sensitivity analysis
sa = sensitivity_analysis(
    explainer, X_train,
    y=y_train,
    depth_range=(3, 5),
    min_samples_leaf_range=(5, 10),
    ccp_alpha_range=(0.0, 0.005),
    n_folds=3,
    random_state=42,
)
print(f"\nsensitivity_analysis: {len(sa.results)} configs tested")
print(f"  Best config: {sa.best_config}")
print(f"  Best fidelity: {sa.best_fidelity:.4f}")

# 13g. Also via Explainer method
sa2 = explainer.sensitivity_analysis(
    X_train, y=y_train,
    depth_range=(3, 5),
    min_samples_leaf_range=(5, 10),
    n_folds=3,
    random_state=42,
)
print(f"  (via Explainer) best fidelity={sa2.best_fidelity:.4f}")

# 13h. Adaptive tolerance
tol = compute_adaptive_tolerance(X_train, feature_names, scale=0.05)
print(f"\ncompute_adaptive_tolerance: {tol}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. Structural stability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("14. Structural stability (compute_structural_stability)")

struct_report = compute_structural_stability(
    explainer, X_train,
    n_bootstraps=10,
    max_depth=4,
    top_k=3,
    random_state=42,
)

print(struct_report)

# Also via Explainer method
struct_report2 = explainer.compute_structural_stability(
    X_train,
    n_bootstraps=10,
    max_depth=4,
    random_state=42,
)
print(f"\n(via Explainer) coverage_overlap={struct_report2.mean_coverage_overlap:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. Complementary metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("15. Complementary metrics (compute_complementary_metrics)")

comp_metrics = explainer.compute_complementary_metrics(result, X_train)

print(comp_metrics)
print(f"\nRule coverage:       {comp_metrics.rule_coverage:.4f}")
print(f"Boundary agreement:  {comp_metrics.boundary_agreement:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 16. Theoretical fidelity bounds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("16. Theoretical fidelity bounds")

# 16a. Full bounds computation
bounds = compute_fidelity_bounds(
    depth=4,
    n_features=4,
    n_samples=len(X_train),
    empirical_infidelity=1.0 - result.report.fidelity,
    delta=0.05,
)

print(bounds)
print(f"\n  VC dimension:             {bounds.vc_dimension}")
print(f"  PAC estimation error:     {bounds.estimation_error_pac:.4f}")
print(f"  Rademacher est. error:    {bounds.estimation_error_rademacher:.4f}")
print(f"  Min fidelity (PAC):       {bounds.min_fidelity_pac:.4f}")
print(f"  Min fidelity (Rademacher): {bounds.min_fidelity_rademacher:.4f}")
print(f"  Sample complexity:        {bounds.sample_complexity_required}")

# 16b. Also via Explainer method
bounds2 = explainer.compute_fidelity_bounds(result, delta=0.05)
print(f"\n(via Explainer) VC={bounds2.vc_dimension}, "
      f"min_fidelity_pac={bounds2.min_fidelity_pac:.4f}")

# 16c. Individual utility functions
vc = vc_dimension_decision_tree(depth=4, n_features=4)
print(f"\nvc_dimension_decision_tree(4, 4) = {vc}")

pac_err = estimation_error_pac(vc_dim=vc, n_samples=len(X_train), delta=0.05)
print(f"estimation_error_pac(vc={vc}, N={len(X_train)}) = {pac_err:.4f}")

rad_err = estimation_error_rademacher(
    depth=4, n_features=4, n_samples=len(X_train), delta=0.05,
)
print(f"estimation_error_rademacher(d=4, p=4, N={len(X_train)}) = {rad_err:.4f}")

n_required = sample_complexity(vc_dim=vc, epsilon=0.05, delta=0.05)
print(f"sample_complexity(vc={vc}, eps=0.05, delta=0.05) = {n_required}")

# 16d. ReLU network analysis
regions = relu_network_regions(n_layers=3, width=32, input_dim=4)
print(f"\nrelu_network_regions(3 layers, width=32, input=4) = {regions}")

d_min = min_depth_for_regions(regions)
print(f"min_depth_for_regions({regions}) = {d_min}")

d_opt = optimal_depth_bound(n_samples=len(X_train), n_features=4)
print(f"optimal_depth_bound(N={len(X_train)}, p=4) = {d_opt}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 17. Counterfactual rule scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("17. Counterfactual rule scoring (score_rules_counterfactual)")

# 17a. Standalone function — score only (no filtering)
cf_report = score_rules_counterfactual(
    result.rules, clf, X_train,
    noise_scale=0.01,
    random_state=42,
)

print(cf_report)
print(f"\nPer-rule scores:")
for rs in cf_report.rule_scores:
    print(f"  Rule {rs.rule_index}: score={rs.score:.2f} "
          f"({rs.n_valid_conditions}/{rs.n_conditions} valid)")

# 17b. With filtering threshold
cf_filtered = score_rules_counterfactual(
    result.rules, clf, X_train,
    validity_threshold=0.5,
    noise_scale=0.01,
    random_state=42,
)

print(f"\nWith threshold=0.5: {cf_filtered.n_rules_retained}/{cf_filtered.n_rules_total} retained")
if cf_filtered.filtered_ruleset is not None:
    print(f"Filtered ruleset has {cf_filtered.filtered_ruleset.num_rules} rules")

# 17c. Via Explainer method
cf_report2 = explainer.score_rules_counterfactual(
    result, X_train,
    validity_threshold=0.5,
    noise_scale=0.01,
    random_state=42,
)
print(f"\n(via Explainer) mean_score={cf_report2.mean_score:.4f}")

# 17d. Inline during extraction
result_cf = explainer.extract_rules(
    X_train, y=y_train,
    max_depth=4,
    counterfactual_validity_threshold=0.5,
    counterfactual_noise_scale=0.01,
)

print(f"\nInline counterfactual:")
print(f"  Rules: {result_cf.rules.num_rules}")
if result_cf.counterfactual_report is not None:
    print(f"  CF report: mean_score={result_cf.counterfactual_report.mean_score:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 18. MDL rule selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("18. MDL rule selection (select_rules_mdl)")

# 18a. Utility functions
print(f"binary_entropy(0.0) = {binary_entropy(0.0):.4f}")
print(f"binary_entropy(0.5) = {binary_entropy(0.5):.4f}")
print(f"binary_entropy(1.0) = {binary_entropy(1.0):.4f}")
print(f"binary_entropy(0.3) = {binary_entropy(0.3):.4f}")

# 18b. Model cost for a rule
if result.rules.rules:
    rule0 = result.rules.rules[0]
    cost = compute_rule_model_cost(
        rule0, n_features=4, n_classes=3, precision_bits=16,
    )
    print(f"\ncompute_rule_model_cost(rule0, p=4, C=3): {cost:.2f} bits")

# 18c. Forward selection
mdl_forward = select_rules_mdl(
    result.rules, clf, X_train,
    n_classes=3,
    precision_bits=16,
    method="forward",
)

print(f"\n{mdl_forward}")

# 18d. Backward elimination
mdl_backward = select_rules_mdl(
    result.rules, clf, X_train,
    n_classes=3,
    method="backward",
)

print(f"\nBackward: {mdl_backward.n_rules_original} -> {mdl_backward.n_rules_selected} rules")

# 18e. Score only
mdl_score = select_rules_mdl(
    result.rules, clf, X_train,
    n_classes=3,
    method="score_only",
)

print(f"\nScore only: {mdl_score.n_rules_selected} rules (all kept)")
for rs in mdl_score.rule_scores:
    print(f"  Rule {rs.rule_index}: model={rs.model_cost:.1f} + data={rs.data_cost:.1f} "
          f"= {rs.total_mdl:.1f} bits (coverage={rs.coverage}, err={rs.error_rate:.2f})")

# 18f. Via Explainer method
mdl_report2 = explainer.select_rules_mdl(
    result, X_train,
    method="forward",
    precision_bits=16,
)
print(f"\n(via Explainer) {mdl_report2.n_rules_original} -> {mdl_report2.n_rules_selected} rules")

# 18g. Inline during extraction
result_mdl = explainer.extract_rules(
    X_train, y=y_train,
    max_depth=4,
    mdl_selection="forward",
    mdl_precision_bits=16,
)

print(f"\nInline MDL selection:")
print(f"  Rules: {result_mdl.rules.num_rules}")
if result_mdl.mdl_report is not None:
    print(f"  MDL reduction: {result_mdl.mdl_report.mdl_reduction:.2f} bits")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 19. Combining counterfactual + MDL inline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("19. Combining counterfactual + MDL in extract_rules")

result_combined = explainer.extract_rules(
    X_train, y=y_train,
    max_depth=5,
    counterfactual_validity_threshold=0.3,
    counterfactual_noise_scale=0.01,
    mdl_selection="forward",
    mdl_precision_bits=16,
)

print(f"Rules: {result_combined.rules.num_rules}")
if result_combined.counterfactual_report:
    print(f"CF mean validity: {result_combined.counterfactual_report.mean_score:.4f}")
if result_combined.mdl_report:
    print(f"MDL reduction: {result_combined.mdl_report.mdl_reduction:.2f} bits")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 20. Working with rules programmatically
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("20. Working with rules programmatically")

# Filter by class
for cls in class_names:
    filtered = result.rules.filter_by_class(cls)
    print(f"  {cls}: {filtered.num_rules} rules")

# Inspect conditions
print("\nFirst rule conditions:")
if result.rules.rules:
    for cond in result.rules.rules[0].conditions:
        print(f"  {cond.feature} {cond.operator} {cond.threshold:.4f}")

# Rule signatures
sigs = result.rules.rule_signatures()
print(f"\nRule signatures: {len(sigs)} unique rules")

# Text export
text = result.rules.to_text()
print(f"\nText export ({len(text)} chars):")
print(text[:300] + "..." if len(text) > 300 else text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 21. Visualization (tree plot + DOT export)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("21. Visualization")

# Save tree plot
result.plot(save_path="/tmp/trace_demo_tree.png", figsize=(16, 8), fontsize=9, dpi=150)
print("Tree plot saved to /tmp/trace_demo_tree.png")

# DOT export
dot_str = result.to_dot()
print(f"DOT export: {len(dot_str)} chars")
print(f"First 200 chars: {dot_str[:200]}...")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 22. Interactive HTML export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("22. Interactive HTML export")

result.to_html("/tmp/trace_demo_report.html")
print("HTML report saved to /tmp/trace_demo_report.html")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 23. SparseObliqueTreeSurrogate — phantom-guided interaction selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("23. SparseObliqueTreeSurrogate — phantom-guided interaction selection")

# Synthetic dataset with a diagonal decision boundary (y = 1 iff x0 + x1 > 2).
# A standard axis-aligned tree produces "phantom splits" here: splits that
# don't correspond to actual black-box boundaries.
# SparseObliqueTreeSurrogate detects those splits via counterfactual probing
# and adds interaction features *only* for the involved feature pairs.

rng_diag = np.random.RandomState(7)
X_diag = rng_diag.uniform(0, 4, size=(800, 2))
y_diag = (X_diag[:, 0] + X_diag[:, 1] > 2).astype(int)
feat_diag = ("x0", "x1")

rf_diag = RandomForestClassifier(n_estimators=50, random_state=7)
rf_diag.fit(X_diag, y_diag)

# 23a. Standalone usage
sparse_surr = SparseObliqueTreeSurrogate(
    max_depth=4,
    min_samples_leaf=5,
    max_iterations=2,
    phantom_threshold=0.3,
    n_probes=20,
    noise_scale=0.01,
    random_state=42,
)
sparse_surr.fit(X_diag, rf_diag.predict(X_diag), model=rf_diag, feature_names=feat_diag)

print(f"Phantom features detected (original indices): {sparse_surr.phantom_features_}")
print(f"Interaction pairs selected (sparse):          {sparse_surr.interaction_pairs_}")
print(f"Iterations performed:                         {sparse_surr.n_iterations_}")
aug_names = sparse_surr.get_augmented_feature_names(feat_diag)
print(f"Augmented feature names:                      {aug_names}")
print(f"Tree depth: {sparse_surr.get_depth()}, leaves: {sparse_surr.get_n_leaves()}")

# 23b. Fidelity comparison: decision_tree vs sparse_oblique_tree vs oblique_tree
from sklearn.metrics import accuracy_score
from trace_xai.surrogates.oblique_tree import ObliqueTreeSurrogate

y_bb_diag = rf_diag.predict(X_diag)

dt_plain = explainer._build_surrogate(max_depth=4, min_samples_leaf=5)
dt_plain.fit(X_diag, y_bb_diag)
fid_dt = accuracy_score(y_bb_diag, dt_plain.predict(X_diag))

full_oblique = ObliqueTreeSurrogate(max_depth=4, min_samples_leaf=5)
full_oblique.fit(X_diag, y_bb_diag)
fid_full = accuracy_score(y_bb_diag, full_oblique.predict(X_diag))

fid_sparse = accuracy_score(y_bb_diag, sparse_surr.predict(X_diag))

n_pairs_full = len(full_oblique._interaction_pairs)
n_pairs_sparse = len(sparse_surr.interaction_pairs_)

print(f"\nFidelity comparison on diagonal boundary:")
print(f"  decision_tree (axis-aligned): {fid_dt:.4f}  | pairs: 0")
print(f"  oblique_tree  (all pairs):    {fid_full:.4f}  | pairs: {n_pairs_full}")
print(f"  sparse_oblique_tree:          {fid_sparse:.4f}  | pairs: {n_pairs_sparse} (sparse)")

# 23c. Via Explainer.extract_rules surrogate_type='sparse_oblique_tree'
explainer_diag = Explainer(rf_diag, feature_names=list(feat_diag))

result_sparse = explainer_diag.extract_rules(
    X_diag,
    surrogate_type="sparse_oblique_tree",
    max_depth=4,
    min_samples_leaf=5,
)

print(f"\nextract_rules(surrogate_type='sparse_oblique_tree'):")
print(f"  Rules: {result_sparse.rules.num_rules}")
print(f"  Fidelity: {result_sparse.report.fidelity:.4f}")
print(f"  Feature names in rules (sample):")
all_feat_in_rules = {
    cond.feature
    for rule in result_sparse.rules.rules
    for cond in rule.conditions
}
for fn in sorted(all_feat_in_rules):
    marker = " *" if " * " in fn else ""
    print(f"    {fn}{marker}")

# 23d. Comparison: surrogate_type='decision_tree' vs 'sparse_oblique_tree'
result_dt = explainer_diag.extract_rules(
    X_diag,
    surrogate_type="decision_tree",
    max_depth=4,
    min_samples_leaf=5,
)

print(f"\nSurrogate comparison on diagonal boundary:")
print(f"  decision_tree:        fidelity={result_dt.report.fidelity:.4f}, "
      f"rules={result_dt.rules.num_rules}")
print(f"  sparse_oblique_tree:  fidelity={result_sparse.report.fidelity:.4f}, "
      f"rules={result_sparse.rules.num_rules}")

# 23e. Fallback mode (no model) — uses feature importance for pair selection
sparse_fallback = SparseObliqueTreeSurrogate(max_depth=4, max_interaction_features=2)
sparse_fallback.fit(X_diag, y_bb_diag)  # no model= kwarg
print(f"\nFallback (no model): pairs={sparse_fallback.interaction_pairs_}, "
      f"phantom_features={sparse_fallback.phantom_features_}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Done
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

section("Done! All trace_xai features demonstrated.")
print("Summary of features covered:")
print("  1.  extract_rules (basic)")
print("  2.  Hold-out fidelity evaluation")
print("  3.  Automatic validation split")
print("  4.  cross_validate_fidelity")
print("  5.  compute_stability (exact + fuzzy)")
print("  6.  compute_confidence_intervals")
print("  7.  Normalised complexity metrics")
print("  8.  PruningConfig (inline + post-hoc)")
print("  9.  Monotonicity (constraints, validation, filtering, enforcement)")
print("  10. extract_stable_rules (ensemble)")
print("  11. Data augmentation (perturbation, boundary, sparse, combined)")
print("  12. Categorical feature decoding")
print("  13. Hyperparameter presets, auto_select_depth, sensitivity_analysis")
print("  14. compute_structural_stability")
print("  15. compute_complementary_metrics")
print("  16. Theoretical fidelity bounds (VC, PAC, Rademacher, sample complexity)")
print("  17. score_rules_counterfactual")
print("  18. select_rules_mdl (forward, backward, score_only)")
print("  19. Combining counterfactual + MDL")
print("  20. Working with rules programmatically")
print("  21. Visualization (tree plot + DOT)")
print("  22. Interactive HTML export")
print("  23. SparseObliqueTreeSurrogate (phantom-guided interaction selection)")
