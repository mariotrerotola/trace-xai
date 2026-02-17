"""Tests for report.py fidelity metrics."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from trace_xai import Explainer
from trace_xai.report import (
    ConfidenceInterval,
    FidelityReport,
    compute_bootstrap_ci,
    compute_fidelity_report,
)


@pytest.fixture()
def iris_data():
    iris = load_iris()
    return iris.data, iris.target, list(iris.feature_names), list(iris.target_names)


class TestFidelityReport:
    def test_report_with_y_true(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=4)
        report = result.report

        assert 0.0 <= report.fidelity <= 1.0
        assert report.accuracy is not None
        assert report.blackbox_accuracy is not None
        assert report.num_rules > 0
        assert report.num_samples == len(X)
        assert isinstance(report.class_fidelity, dict)
        assert len(report.class_fidelity) > 0

    def test_report_without_y_true(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, max_depth=4)
        report = result.report

        assert 0.0 <= report.fidelity <= 1.0
        assert report.accuracy is None
        assert report.blackbox_accuracy is None

    def test_report_str(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=3)
        text = str(result.report)

        assert "Fidelity Report" in text
        assert "Fidelity" in text
        assert "Per-class fidelity" in text

    def test_report_has_evaluation_type(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=3)
        assert result.report.evaluation_type == "in_sample"
        assert "Evaluation type:" in str(result.report)


class TestBootstrapCI:
    def test_ci_bounds(self):
        rng = np.random.RandomState(42)
        values = rng.normal(0.9, 0.02, size=500)
        ci = compute_bootstrap_ci(values, confidence_level=0.95)

        assert isinstance(ci, ConfidenceInterval)
        assert ci.lower <= ci.point_estimate <= ci.upper
        assert ci.confidence_level == 0.95

    def test_ci_narrow_with_constant(self):
        values = np.full(100, 0.95)
        ci = compute_bootstrap_ci(values)
        assert abs(ci.lower - 0.95) < 1e-9
        assert abs(ci.upper - 0.95) < 1e-9
