#!/usr/bin/env python3
"""
Explainability audit for a credit scoring model using TRACE.

This example simulates a realistic regulatory scenario: a bank deploys a
GradientBoosting classifier to approve/reject loan applications and must
provide a human-readable explanation of how the model makes decisions.

TRACE extracts IF-THEN rules from the black-box using the new
SparseObliqueTreeSurrogate, which detects "phantom splits" (axis-aligned
splits that the black-box does not actually use) via counterfactual probing
and adds interaction features only for those feature pairs — yielding more
faithful explanations with fewer spurious conditions than either a plain
decision tree or a full oblique tree.

The pipeline also validates stability, verifies rule boundaries via
counterfactual scoring, selects the most informative rules via MDL, and
enforces monotonicity constraints required by financial regulators.

Dataset: German Credit (UCI), 1 000 applicants, 20 features.
"""

import warnings

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from trace_xai import (
    Explainer,
    PruningConfig,
    SparseObliqueTreeSurrogate,
    # Monotonicity
    validate_monotonicity,
    enforce_monotonicity,
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


# ── 1. Load and prepare German Credit data ───────────────────────────────

print("=" * 70)
print("  TRACE Credit Scoring Audit — SparseObliqueTreeSurrogate")
print("=" * 70)

print("\n1. Loading German Credit dataset ...")

credit = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
df = credit.frame

# Target: "good" = approved, "bad" = rejected
y_raw = (df["class"] == "good").astype(int).values
class_names = ["rejected", "approved"]

# Select a meaningful subset of features for readability
selected_features = [
    "duration",                # Loan duration in months
    "credit_amount",           # Loan amount
    "installment_commitment",  # Installment rate (% of disposable income)
    "age",                     # Age in years
    "num_dependents",          # Number of dependents
    "existing_credits",        # Number of existing credits at this bank
    "residence_since",         # Years at current residence
    "employment",              # Employment duration (categorical)
    "personal_status",         # Sex & marital status (categorical)
    "housing",                 # Own / rent / free (categorical)
]

df_sel = df[selected_features].copy()

cat_cols = ["employment", "personal_status", "housing"]
num_cols = [c for c in selected_features if c not in cat_cols]

readable_names = {
    "duration": "loan_duration_months",
    "credit_amount": "loan_amount",
    "installment_commitment": "installment_rate_pct",
    "age": "applicant_age",
    "num_dependents": "n_dependents",
    "existing_credits": "n_existing_credits",
    "residence_since": "years_at_residence",
    "employment": "employment_duration",
    "personal_status": "personal_status",
    "housing": "housing_type",
}

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ],
    remainder="drop",
)

X_processed = preprocessor.fit_transform(df_sel)
feature_names = [readable_names[c] for c in num_cols + cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_raw, test_size=0.3, random_state=42, stratify=y_raw,
)

print(f"   Samples: {len(y_raw)} (train={len(y_train)}, test={len(y_test)})")
print(f"   Features: {feature_names}")
print(f"   Classes: {class_names}")
print(f"   Class balance: approved={y_raw.mean():.1%}, rejected={1 - y_raw.mean():.1%}")


# ── 2. Train the black-box model ─────────────────────────────────────────

print("\n2. Training GradientBoosting classifier ...")

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42,
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"   Black-box accuracy: train={train_acc:.4f}, test={test_acc:.4f}")


# ── 3. Create the TRACE explainer ────────────────────────────────────────

explainer = Explainer(
    model,
    feature_names=feature_names,
    class_names=class_names,
)


# ── 4. Find the best surrogate depth automatically ───────────────────────

print("\n3. Auto-selecting surrogate depth (target fidelity >= 0.85) ...")

depth_result = auto_select_depth(
    explainer, X_train,
    y=y_train,
    target_fidelity=0.85,
    min_depth=3,
    max_depth=6,
    n_folds=5,
    random_state=42,
)

print(f"   Selected depth: {depth_result.best_depth}")
print(f"   CV fidelity at that depth: {depth_result.selected_fidelity:.4f}")
print(f"   All depths tested:")
for d, f in sorted(depth_result.fidelity_scores.items()):
    marker = " <-- selected" if d == depth_result.best_depth else ""
    print(f"     depth={d}: fidelity={f:.4f}{marker}")

best_depth = depth_result.best_depth


# ── 5. Surrogate comparison: DT vs Sparse vs Full oblique ────────────────

print(f"\n4. Surrogate comparison at depth={best_depth} ...")

y_bb_train = model.predict(X_train)

# 4a. decision_tree baseline
result_dt = explainer.extract_rules(X_train, y=y_train, max_depth=best_depth,
                                    surrogate_type="decision_tree")
fid_dt = result_dt.report.fidelity

# 4b. SparseObliqueTreeSurrogate (standalone, for inspection)
sparse_surr = SparseObliqueTreeSurrogate(
    max_depth=best_depth,
    min_samples_leaf=5,
    max_iterations=2,
    phantom_threshold=0.3,
    n_probes=20,
    noise_scale=0.05,
    random_state=42,
)
sparse_surr.fit(X_train, y_bb_train, model=model, feature_names=tuple(feature_names))
fid_sparse_standalone = accuracy_score(y_bb_train, sparse_surr.predict(X_train))

# 4c. Full oblique_tree (all pairs) for reference
n_feats = X_train.shape[1]
full_surr = ObliqueTreeSurrogate(max_depth=best_depth, min_samples_leaf=5)
full_surr.fit(X_train, y_bb_train)
fid_full = accuracy_score(y_bb_train, full_surr.predict(X_train))

n_pairs_full = len(full_surr._interaction_pairs)   # n*(n-1)/2
n_pairs_sparse = len(sparse_surr.interaction_pairs_)

print(f"\n   {'Surrogate':<22} {'Fidelity':>10} {'Pairs':>8} {'Notes'}")
print(f"   {'─' * 60}")
print(f"   {'decision_tree':<22} {fid_dt:>10.4f} {'0':>8}  axis-aligned only")
print(f"   {'sparse_oblique_tree':<22} {fid_sparse_standalone:>10.4f} "
      f"{n_pairs_sparse:>8}  phantom-guided (sparse)")
print(f"   {'oblique_tree':<22} {fid_full:>10.4f} "
      f"{n_pairs_full:>8}  all {n_feats}*({n_feats}-1)/2 pairs")

print(f"\n   Phantom features detected: {sparse_surr.phantom_features_}")
print(f"   Interaction pairs selected: {sparse_surr.interaction_pairs_}")
print(f"   Augmented feature names: {sparse_surr.get_augmented_feature_names(tuple(feature_names))}")


# ── 6. Augment training data for better surrogate coverage ───────────────

print("\n5. Augmenting training data (perturbation) ...")

X_aug, y_aug = augment_data(
    X_train, model,
    strategy="perturbation",
    n_neighbors=3,
    noise_scale=0.05,
    random_state=42,
)

print(f"   Original samples: {len(X_train)}")
print(f"   Augmented samples: {len(X_aug)}")


# ── 7. Extract rules via SparseObliqueTreeSurrogate + full pipeline ───────

print(f"\n6. Extracting rules with sparse_oblique_tree (depth={best_depth}, pruning, CF + MDL) ...")

# Step A: extract with pruning
result_raw = explainer.extract_rules(
    X_aug, y=y_aug,
    max_depth=best_depth,
    surrogate_type="sparse_oblique_tree",
    X_val=X_test,
    y_val=y_test,
    pruning=PruningConfig(
        min_confidence=0.55,
        min_samples=10,
        remove_redundant=True,
    ),
)

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
    ruleset_after_cf, model, X_aug,
    n_classes=2,
    method="forward",
    precision_bits=8,
)

result = result_raw
if mdl_report.selected_ruleset.num_rules > 0:
    final_rules = mdl_report.selected_ruleset
else:
    final_rules = ruleset_after_cf

# Report the interaction pairs used in the final surrogate
final_sparse_surr = result.surrogate
print(f"\n   Sparse surrogate — interaction pairs: {final_sparse_surr.interaction_pairs_}")
print(f"   Sparse surrogate — phantom features:   {final_sparse_surr.phantom_features_}")
print(f"   Sparse surrogate — iterations:         {final_sparse_surr.n_iterations_}")

print(f"\n   --- Fidelity Report (hold-out) ---")
print(f"   Fidelity: {result.report.fidelity:.4f}")
print(f"   Surrogate accuracy: {result.report.accuracy:.4f}")
print(f"   Raw rules: {result.rules.num_rules}")
print(f"   Avg rule length: {result.report.avg_rule_length:.1f}")

print(f"\n   --- Counterfactual Validity ---")
print(f"   Mean validity score: {cf_report.mean_score:.4f} (± {cf_report.std_score:.4f})")
print(f"   Rules retained after CF: {cf_report.n_rules_retained}/{cf_report.n_rules_total}")

print(f"\n   --- MDL Selection ---")
print(f"   Rules after MDL: {mdl_report.n_rules_selected}")
print(f"   MDL reduction: {mdl_report.mdl_reduction:.1f} bits")
print(f"   Final rules: {final_rules.num_rules}")


# ── 8. Print the extracted rules ─────────────────────────────────────────

print("\n7. Final rules (sparse_oblique_tree, after CF + MDL):")
print("-" * 70)
print(final_rules)


# ── 9. Monotonicity audit ────────────────────────────────────────────────

print("\n8. Monotonicity audit ...")

constraints = {
    "applicant_age": 1,           # older → more likely approved
    "installment_rate_pct": -1,   # higher rate → less likely approved
}

mono_report = validate_monotonicity(final_rules, constraints)
print(f"   Constraints: {constraints}")
print(f"   Compliant: {mono_report.is_compliant}")
print(f"   Violations: {len(mono_report.violations)}")

if not mono_report.is_compliant:
    for v in mono_report.violations[:5]:
        print(f"     - {v.description}")

    enforcement = enforce_monotonicity(
        final_rules, constraints,
        surrogate=result.surrogate,
        X=X_test,
        model=model,
    )
    print(f"   After enforcement: {enforcement.rules_removed} rules removed, "
          f"fidelity impact: {enforcement.fidelity_impact}")


# ── 10. Structural stability ─────────────────────────────────────────────

print("\n9. Structural stability (sparse_oblique_tree) ...")

struct = compute_structural_stability(
    explainer, X_train,
    n_bootstraps=15,
    max_depth=best_depth,
    random_state=42,
)

print(f"   Coverage overlap:         {struct.mean_coverage_overlap:.4f} ± {struct.std_coverage_overlap:.4f}")
print(f"   Prediction agreement:     {struct.mean_prediction_agreement:.4f} ± {struct.std_prediction_agreement:.4f}")


# ── 11. Cross-validated fidelity ─────────────────────────────────────────

print("\n10. Cross-validated fidelity ...")

cv_report = explainer.cross_validate_fidelity(
    X_train, y=y_train,
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
    valid_str = ", ".join(
        cond.feature for cv in rs.condition_validities if cv.is_valid
        for cond in [cv.condition]
    )
    phantom_str = ", ".join(
        cond.feature for cv in rs.condition_validities if not cv.is_valid
        for cond in [cv.condition]
    )
    print(f"   Rule {rs.rule_index}: score={rs.score:.2f} "
          f"| valid=[{valid_str}] | phantom=[{phantom_str}]")


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


# ── 15. Export audit artefacts ────────────────────────────────────────────

print("\n14. Exporting audit artefacts ...")

result.to_html("/tmp/credit_audit_report.html")
print("   HTML report: /tmp/credit_audit_report.html")

result.plot(save_path="/tmp/credit_surrogate_tree.png", figsize=(20, 10), dpi=150)
print("   Tree plot:   /tmp/credit_surrogate_tree.png")

dot = result.to_dot()
with open("/tmp/credit_surrogate_tree.dot", "w") as f:
    f.write(dot)
print("   DOT export:  /tmp/credit_surrogate_tree.dot")

rules_text = final_rules.to_text()
with open("/tmp/credit_rules.txt", "w") as f:
    f.write(rules_text)
print("   Rules text:  /tmp/credit_rules.txt")


# ── 16. Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("  Audit Summary")
print(f"{'=' * 70}")
print(f"  Black-box model:           GradientBoosting (100 trees, depth=3)")
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
print(f"  Structural stability:")
print(f"    Coverage overlap:        {struct.mean_coverage_overlap:.4f}")
print(f"    Prediction agreement:    {struct.mean_prediction_agreement:.4f}")
print(f"{'=' * 70}")
