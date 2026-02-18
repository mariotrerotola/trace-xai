#!/usr/bin/env python3
"""
Explainability audit for an income prediction model using TRACE.

This example simulates a regulatory scenario: an organisation deploys a
GradientBoosting classifier to predict whether an individual earns more
than $50K/year.  TRACE extracts human-readable IF-THEN rules using the
SparseObliqueTreeSurrogate, which avoids phantom splits — axis-aligned
boundaries that the black-box model does not actually use — by adding
interaction features only where counterfactual probing reveals that a
plain axis-aligned split fails to reflect real model behaviour.

The pipeline also decodes categorical features, validates each split
boundary via counterfactual probing, selects information-efficient rules
via MDL, and checks monotonicity constraints.

Dataset: Adult Census (UCI), ~48 000 individuals, 14 features.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from trace_xai import (
    Explainer,
    PruningConfig,
    SparseObliqueTreeSurrogate,
    CategoricalMapping,
    decode_ruleset,
    # Monotonicity
    validate_monotonicity,
    # Augmentation
    augment_data,
    # Hyperparameters
    auto_select_depth,
    # Stability
    compute_structural_stability,
    # Counterfactual & MDL
    score_rules_counterfactual,
    select_rules_mdl,
)
from trace_xai.surrogates.oblique_tree import ObliqueTreeSurrogate


# ── 1. Load and prepare Adult Census data ─────────────────────────────────

print("=" * 72)
print("  TRACE Income Prediction Audit — SparseObliqueTreeSurrogate")
print("=" * 72)

print("\n1. Loading Adult Census dataset ...")

adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
df = adult.frame.dropna()

y_raw = (df["class"].str.strip() == ">50K").astype(int).values
class_names = ["<=50K", ">50K"]

num_features = ["age", "education-num", "hours-per-week",
                "capital-gain", "capital-loss"]
cat_features = ["workclass", "marital-status", "occupation", "relationship"]

all_features = num_features + cat_features

cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", cat_encoder, cat_features),
    ],
    remainder="drop",
)

X_processed = preprocessor.fit_transform(df[all_features])
feature_names = num_features + cat_features

# CategoricalMappings for decoding rules back to readable form
fitted_cat_encoder = preprocessor.named_transformers_["cat"]
cat_mappings = []
for i, feat in enumerate(cat_features):
    cats = fitted_cat_encoder.categories_[i]
    cat_mappings.append(CategoricalMapping(
        original_name=feat,
        encoding="ordinal",
        encoded_columns=(feat,),
        categories=tuple(str(c) for c in cats),
    ))

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_raw, test_size=0.2, random_state=42, stratify=y_raw,
)

print(f"   Samples: {len(y_raw):,} (train={len(y_train):,}, test={len(y_test):,})")
print(f"   Features: {feature_names}")
print(f"   Classes: {class_names}")
print(f"   Class balance: >50K={y_raw.mean():.1%}, <=50K={1 - y_raw.mean():.1%}")


# ── 2. Train the black-box model ─────────────────────────────────────────

print("\n2. Training GradientBoosting classifier ...")

model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"   Black-box accuracy: train={train_acc:.4f}, test={test_acc:.4f}")


# ── 3. Create TRACE explainer ────────────────────────────────────────────

explainer = Explainer(
    model,
    feature_names=feature_names,
    class_names=class_names,
)


# ── 4. Auto-select surrogate depth ───────────────────────────────────────

print("\n3. Auto-selecting surrogate depth (target fidelity >= 0.91) ...")

depth_result = auto_select_depth(
    explainer, X_train,
    y=y_train,
    target_fidelity=0.91,
    min_depth=3,
    max_depth=6,
    n_folds=5,
    random_state=42,
)

best_depth = depth_result.best_depth
print(f"   Selected depth: {best_depth}")
print(f"   CV fidelity at that depth: {depth_result.selected_fidelity:.4f}")
for d, f in sorted(depth_result.fidelity_scores.items()):
    marker = " <-- selected" if d == best_depth else ""
    print(f"     depth={d}: fidelity={f:.4f}{marker}")


# ── 5. Surrogate comparison on a subsample ───────────────────────────────

print(f"\n4. Surrogate comparison at depth={best_depth} (subsample 5 000 pts) ...")

rng = np.random.RandomState(42)
cmp_idx = rng.choice(len(X_train), size=min(5000, len(X_train)), replace=False)
X_cmp = X_train[cmp_idx]
y_cmp = y_train[cmp_idx]
y_bb_cmp = model.predict(X_cmp)

# decision_tree baseline
result_dt = explainer.extract_rules(X_cmp, y=y_cmp, max_depth=best_depth,
                                    surrogate_type="decision_tree")
fid_dt = result_dt.report.fidelity

# SparseObliqueTreeSurrogate (standalone for inspection)
sparse_surr = SparseObliqueTreeSurrogate(
    max_depth=best_depth,
    min_samples_leaf=10,
    max_iterations=2,
    phantom_threshold=0.3,
    n_probes=20,
    noise_scale=0.05,
    random_state=42,
)
sparse_surr.fit(X_cmp, y_bb_cmp, model=model, feature_names=tuple(feature_names))
fid_sparse = accuracy_score(y_bb_cmp, sparse_surr.predict(X_cmp))

# Full oblique_tree (all pairs) for reference
full_surr = ObliqueTreeSurrogate(max_depth=best_depth, min_samples_leaf=10)
full_surr.fit(X_cmp, y_bb_cmp)
fid_full = accuracy_score(y_bb_cmp, full_surr.predict(X_cmp))

n_feats = X_cmp.shape[1]
n_pairs_sparse = len(sparse_surr.interaction_pairs_)
n_pairs_full = len(full_surr._interaction_pairs)

print(f"\n   {'Surrogate':<22} {'Fidelity':>10} {'Pairs':>8}  Notes")
print(f"   {'─' * 62}")
print(f"   {'decision_tree':<22} {fid_dt:>10.4f} {'0':>8}  axis-aligned only")
print(f"   {'sparse_oblique_tree':<22} {fid_sparse:>10.4f} "
      f"{n_pairs_sparse:>8}  phantom-guided (sparse)")
print(f"   {'oblique_tree':<22} {fid_full:>10.4f} "
      f"{n_pairs_full:>8}  all {n_feats}*({n_feats}-1)/2 pairs")
print(f"\n   Phantom features detected: {sparse_surr.phantom_features_}")
print(f"   Interaction pairs selected: {sparse_surr.interaction_pairs_}")


# ── 6. Augment training data ─────────────────────────────────────────────

print("\n5. Augmenting training data (perturbation) ...")

aug_idx = rng.choice(len(X_train), size=min(5000, len(X_train)), replace=False)
X_aug_base = X_train[aug_idx]

X_aug, y_aug = augment_data(
    X_aug_base, model,
    strategy="perturbation",
    n_neighbors=2,
    noise_scale=0.05,
    random_state=42,
)

print(f"   Base samples: {len(X_aug_base):,}")
print(f"   Augmented samples: {len(X_aug):,}")


# ── 7. Extract rules via sparse_oblique_tree + full pipeline ─────────────

print(f"\n6. Extracting rules with sparse_oblique_tree (depth={best_depth}, pruning, CF + MDL) ...")

# Step A: extract with pruning
result_raw = explainer.extract_rules(
    X_aug, y=y_aug,
    max_depth=best_depth,
    surrogate_type="sparse_oblique_tree",
    X_val=X_test, y_val=y_test,
    pruning=PruningConfig(
        min_confidence=0.60,
        min_samples=20,
        remove_redundant=True,
    ),
)

# Report the interaction pairs used
final_sparse_surr = result_raw.surrogate
print(f"\n   Sparse surrogate — interaction pairs: {final_sparse_surr.interaction_pairs_}")
print(f"   Sparse surrogate — phantom features:   {final_sparse_surr.phantom_features_}")
print(f"   Sparse surrogate — iterations:         {final_sparse_surr.n_iterations_}")

# Step B: counterfactual scoring
cf_report = score_rules_counterfactual(
    result_raw.rules, model, X_test,
    validity_threshold=0.3,
    noise_scale=0.1,
    n_probes=30,
    random_state=42,
)

if cf_report.filtered_ruleset and cf_report.filtered_ruleset.num_rules > 0:
    ruleset_after_cf = cf_report.filtered_ruleset
else:
    ruleset_after_cf = result_raw.rules

# Step C: MDL selection
mdl_report = select_rules_mdl(
    ruleset_after_cf, model, X_test,
    n_classes=2,
    method="forward",
    precision_bits=8,
)

if mdl_report.selected_ruleset.num_rules > 0:
    final_rules = mdl_report.selected_ruleset
else:
    final_rules = ruleset_after_cf

result = result_raw

print(f"\n   --- Fidelity Report (hold-out) ---")
print(f"   Fidelity: {result.report.fidelity:.4f}")
print(f"   Surrogate accuracy: {result.report.accuracy:.4f}")
print(f"   Raw rules: {result.rules.num_rules}")
print(f"   Avg rule length: {result.report.avg_rule_length:.1f}")

print(f"\n   --- Counterfactual Validity ---")
print(f"   Mean validity score: {cf_report.mean_score:.4f} (± {cf_report.std_score:.4f})")
print(f"   Rules retained after CF: {cf_report.n_rules_retained}/{cf_report.n_rules_total}")
print(f"   Per-rule scores: {', '.join(f'{rs.score:.2f}' for rs in cf_report.rule_scores)}")

print(f"\n   --- MDL Selection ---")
print(f"   Rules after MDL: {mdl_report.n_rules_selected}")
print(f"   MDL reduction: {mdl_report.mdl_reduction:.1f} bits")
print(f"   Final rules: {final_rules.num_rules}")


# ── 8. Decode categorical features and print rules ───────────────────────

print(f"\n7. Final rules (decoded, sparse_oblique_tree, after CF + MDL):")
print("-" * 72)

decoded_rules = decode_ruleset(final_rules, cat_mappings)
print(decoded_rules)


# ── 9. Monotonicity audit ────────────────────────────────────────────────

print("\n8. Monotonicity audit ...")

constraints = {
    "age": 1,               # older → more likely high income
    "education-num": 1,     # more education → more likely high income
    "hours-per-week": 1,    # more hours → more likely high income
}

mono_report = validate_monotonicity(final_rules, constraints)
print(f"   Constraints: {constraints}")
print(f"   Compliant: {mono_report.is_compliant}")
print(f"   Violations: {len(mono_report.violations)}")

if not mono_report.is_compliant:
    for v in mono_report.violations[:5]:
        print(f"     - {v.description}")


# ── 10. Structural stability ─────────────────────────────────────────────

print("\n9. Structural stability ...")

struct = compute_structural_stability(
    explainer, X_train[:5000],
    n_bootstraps=15,
    max_depth=best_depth,
    random_state=42,
)

print(f"   Coverage overlap:       {struct.mean_coverage_overlap:.4f} ± {struct.std_coverage_overlap:.4f}")
print(f"   Prediction agreement:   {struct.mean_prediction_agreement:.4f} ± {struct.std_prediction_agreement:.4f}")


# ── 11. Cross-validated fidelity ─────────────────────────────────────────

print("\n10. Cross-validated fidelity ...")

cv_report = explainer.cross_validate_fidelity(
    X_train[:5000], y=y_train[:5000],
    n_folds=5,
    max_depth=best_depth,
    random_state=42,
)

print(f"   {cv_report}")


# ── 12. Confidence intervals ─────────────────────────────────────────────

print("\n11. Bootstrap confidence intervals (95%) ...")

cis = explainer.compute_confidence_intervals(
    result, X_test,
    y=y_test,
    n_bootstraps=1000,
    confidence_level=0.95,
    random_state=42,
)

for metric, ci in cis.items():
    print(f"   {metric}: {ci.point_estimate:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")


# ── 13. Detailed counterfactual analysis ─────────────────────────────────

print("\n12. Detailed counterfactual analysis on final rules ...")

cf_detailed = score_rules_counterfactual(
    final_rules, model, X_test,
    noise_scale=0.1,
    n_probes=30,
    random_state=42,
)

for rs in cf_detailed.rule_scores:
    valid_feats = [cv.condition.feature for cv in rs.condition_validities if cv.is_valid]
    phantom_feats = [cv.condition.feature for cv in rs.condition_validities if not cv.is_valid]
    print(f"   Rule {rs.rule_index}: score={rs.score:.2f} "
          f"| valid={valid_feats} | phantom={phantom_feats}")


# ── 14. Detailed MDL analysis ────────────────────────────────────────────

print("\n13. Detailed MDL cost breakdown ...")

mdl_detailed = select_rules_mdl(
    final_rules, model, X_test,
    n_classes=2,
    method="score_only",
)

print(f"   {'Rule':>6}  {'Model':>8}  {'Data':>8}  {'Total':>8}  "
      f"{'Coverage':>8}  {'Error':>8}")
print(f"   {'-' * 56}")
for rs in mdl_detailed.rule_scores:
    print(f"   {rs.rule_index:>6}  {rs.model_cost:>8.1f}  {rs.data_cost:>8.1f}  "
          f"{rs.total_mdl:>8.1f}  {rs.coverage:>8}  {rs.error_rate:>8.2f}")


# ── 15. Export ────────────────────────────────────────────────────────────

print("\n14. Exporting audit artefacts ...")

result.to_html("/tmp/income_audit_report.html")
print("   HTML report: /tmp/income_audit_report.html")

result.plot(save_path="/tmp/income_surrogate_tree.png", figsize=(24, 12), dpi=150)
print("   Tree plot:   /tmp/income_surrogate_tree.png")

rules_text = decoded_rules.to_text()
with open("/tmp/income_rules.txt", "w") as f:
    f.write(rules_text)
print("   Rules text:  /tmp/income_rules.txt")


# ── 16. Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 72}")
print("  Audit Summary")
print(f"{'=' * 72}")
print(f"  Black-box model:           GradientBoosting (200 trees, depth=5)")
print(f"  Black-box test accuracy:   {test_acc:.4f}")
print(f"  Surrogate type:            sparse_oblique_tree")
print(f"  Surrogate depth:           {best_depth} (auto-selected)")
print(f"  Phantom features detected: {sparse_surr.phantom_features_}")
print(f"  Interaction pairs (sparse):{n_pairs_sparse} / {n_pairs_full} (full oblique)")
print(f"  Hold-out fidelity:         {result.report.fidelity:.4f}")
print(f"  CV fidelity:               {cv_report.mean_fidelity:.4f} ± {cv_report.std_fidelity:.4f}")
fid_ci = cis["fidelity"]
print(f"  Fidelity 95% CI:           [{fid_ci.lower:.4f}, {fid_ci.upper:.4f}]")
print(f"  Final rules:               {final_rules.num_rules}")
print(f"  Monotonicity compliant:    {mono_report.is_compliant}")
print(f"  CF mean validity:          {cf_report.mean_score:.4f}")
cf_scores_str = ", ".join(f"{rs.score:.2f}" for rs in cf_report.rule_scores)
print(f"  CF per-rule scores:        [{cf_scores_str}]")
print(f"  Structural stability:")
print(f"    Coverage overlap:        {struct.mean_coverage_overlap:.4f}")
print(f"    Prediction agreement:    {struct.mean_prediction_agreement:.4f}")
print(f"{'=' * 72}")
