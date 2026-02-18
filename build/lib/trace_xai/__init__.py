"""TRACE (trace_xai) - Tree-based Rule Approximation for Comprehensible Explanations."""

from .explainer import ExplanationResult, Explainer
from .report import (
    CVFidelityReport,
    ConfidenceInterval,
    FidelityReport,
    StabilityReport,
    compute_bootstrap_ci,
    compute_fidelity_report,
    compute_regression_fidelity_report,
)
from .ruleset import Condition, Rule, RuleSet
from .html_export import export_html
from .visualization import export_dot, plot_surrogate_tree
from .pruning import PruningConfig, PruningReport, prune_ruleset
from .monotonicity import (
    MonotonicityReport,
    MonotonicityViolation,
    MonotonicityEnforcementResult,
    validate_monotonicity,
    filter_monotonic_violations,
    enforce_monotonicity,
)
from .ensemble import (
    EnsembleReport,
    StableRule,
    extract_ensemble_rules_adaptive,
    rank_rules_by_frequency,
)
from .augmentation import (
    augment_data,
    perturbation_augmentation,
    boundary_augmentation,
    sparse_region_augmentation,
)
from .categorical import (
    CategoricalMapping,
    CategoricalCondition,
    decode_ruleset,
    decode_conditions,
)
from .hyperparams import (
    HyperparamPreset,
    AutoDepthResult,
    SensitivityResult,
    get_preset,
    auto_select_depth,
    sensitivity_analysis,
    compute_adaptive_tolerance,
    PRESETS,
)
from .stability import (
    StructuralStabilityReport,
    compute_structural_stability,
)
from .metrics import (
    ComplementaryMetrics,
    compute_complementary_metrics,
)
from .theoretical_bounds import (
    FidelityBound,
    compute_fidelity_bounds,
    vc_dimension_decision_tree,
    estimation_error_pac,
    estimation_error_rademacher,
    sample_complexity,
    relu_network_regions,
    min_depth_for_regions,
    optimal_depth_bound,
)

__all__ = [
    "Explainer",
    "ExplanationResult",
    "Condition",
    "Rule",
    "RuleSet",
    "FidelityReport",
    "CVFidelityReport",
    "StabilityReport",
    "ConfidenceInterval",
    "compute_fidelity_report",
    "compute_regression_fidelity_report",
    "compute_bootstrap_ci",
    "plot_surrogate_tree",
    "export_dot",
    "export_html",
    # Pruning
    "PruningConfig",
    "PruningReport",
    "prune_ruleset",
    # Monotonicity
    "MonotonicityReport",
    "MonotonicityViolation",
    "MonotonicityEnforcementResult",
    "validate_monotonicity",
    "filter_monotonic_violations",
    "enforce_monotonicity",
    # Ensemble
    "EnsembleReport",
    "StableRule",
    "extract_ensemble_rules_adaptive",
    "rank_rules_by_frequency",
    # Augmentation
    "augment_data",
    "perturbation_augmentation",
    "boundary_augmentation",
    "sparse_region_augmentation",
    # Categorical
    "CategoricalMapping",
    "CategoricalCondition",
    "decode_ruleset",
    "decode_conditions",
    # Hyperparameters
    "HyperparamPreset",
    "AutoDepthResult",
    "SensitivityResult",
    "get_preset",
    "auto_select_depth",
    "sensitivity_analysis",
    "compute_adaptive_tolerance",
    "PRESETS",
    # Structural stability
    "StructuralStabilityReport",
    "compute_structural_stability",
    # Complementary metrics
    "ComplementaryMetrics",
    "compute_complementary_metrics",
    # Theoretical bounds
    "FidelityBound",
    "compute_fidelity_bounds",
    "vc_dimension_decision_tree",
    "estimation_error_pac",
    "estimation_error_rademacher",
    "sample_complexity",
    "relu_network_regions",
    "min_depth_for_regions",
    "optimal_depth_bound",
]
