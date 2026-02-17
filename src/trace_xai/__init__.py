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
]
