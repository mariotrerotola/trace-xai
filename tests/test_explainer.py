"""Tests for explainer.py - Explainer and ExplanationResult."""

import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from trace_xai import Explainer, ExplanationResult, RuleSet
from trace_xai.report import CVFidelityReport, FidelityReport, StabilityReport


@pytest.fixture()
def iris_data():
    iris = load_iris()
    return iris.data, iris.target, list(iris.feature_names), list(iris.target_names)


class TestExplainerValidation:
    def test_no_predict_raises(self):
        """A model without .predict() must raise TypeError."""
        with pytest.raises(TypeError, match="predict"):
            Explainer(object(), feature_names=["f1"], class_names=["A"])

    def test_plain_class_without_predict(self):
        class BadModel:
            pass

        with pytest.raises(TypeError, match="predict"):
            Explainer(BadModel(), feature_names=["f1"], class_names=["A"])


class TestExplainerEndToEnd:
    """End-to-end tests with real sklearn classifiers."""

    @pytest.mark.parametrize(
        "model_cls,kwargs",
        [
            (RandomForestClassifier, {"n_estimators": 10, "random_state": 0}),
            (MLPClassifier, {"hidden_layer_sizes": (16,), "max_iter": 300, "random_state": 0}),
            (SVC, {"kernel": "rbf", "random_state": 0}),
        ],
        ids=["RandomForest", "MLP", "SVM"],
    )
    def test_extract_rules(self, iris_data, model_cls, kwargs):
        X, y, feat, cls = iris_data
        model = model_cls(**kwargs).fit(X, y)

        explainer = Explainer(model, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=4)

        assert isinstance(result, ExplanationResult)
        assert isinstance(result.rules, RuleSet)
        assert isinstance(result.report, FidelityReport)
        assert result.rules.num_rules > 0
        assert result.report.fidelity > 0.5  # surrogate should be reasonably faithful

    def test_max_depth_controls_complexity(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)

        shallow = explainer.extract_rules(X, max_depth=2)
        deep = explainer.extract_rules(X, max_depth=6)

        assert shallow.report.surrogate_depth <= 2
        assert deep.report.surrogate_depth <= 6
        assert shallow.rules.num_rules <= deep.rules.num_rules

    def test_rule_format(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, max_depth=3)

        for rule in result.rules.rules:
            text = str(rule)
            assert text.startswith("IF ")
            assert "THEN class =" in text
            assert "confidence=" in text
            assert "samples=" in text

    def test_plot_saves_file(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, max_depth=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tree.png")
            result.plot(save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_to_dot(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, max_depth=3)

        dot = result.to_dot()
        assert "digraph" in dot
        assert "label=" in dot

    def test_str(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=3)

        text = str(result)
        assert "Rule 1:" in text
        assert "Fidelity Report" in text


class TestHoldOutFidelity:
    """Tests for hold-out and validation_split evaluation."""

    def test_default_is_in_sample(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=4)
        assert result.report.evaluation_type == "in_sample"
        assert result.train_report is None

    def test_x_val(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(
            X[:100], y=y[:100], X_val=X[100:], y_val=y[100:],
        )
        assert result.report.evaluation_type == "hold_out"
        assert result.train_report is not None
        assert result.train_report.evaluation_type == "in_sample"

    def test_validation_split(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, validation_split=0.3)
        assert result.report.evaluation_type == "validation_split"
        assert result.train_report is not None

    def test_xval_and_split_raises(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        with pytest.raises(ValueError, match="mutually exclusive"):
            explainer.extract_rules(X, X_val=X, validation_split=0.3)


class TestCrossValidatedFidelity:
    def test_cv_3_fold_iris(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        cv_report = explainer.cross_validate_fidelity(X, y=y, n_folds=3)

        assert isinstance(cv_report, CVFidelityReport)
        assert cv_report.n_folds == 3
        assert len(cv_report.fold_reports) == 3
        assert 0 < cv_report.mean_fidelity <= 1.0
        assert cv_report.std_fidelity >= 0.0
        assert cv_report.mean_accuracy is not None

    def test_cv_without_y(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        cv_report = explainer.cross_validate_fidelity(X, n_folds=3)
        assert cv_report.mean_accuracy is None


class TestStabilityScore:
    def test_stability(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        report = explainer.compute_stability(X, n_bootstraps=5)

        assert isinstance(report, StabilityReport)
        assert 0.0 <= report.mean_jaccard <= 1.0
        assert report.n_bootstraps == 5
        assert len(report.pairwise_jaccards) == 10  # C(5,2)

    def test_stability_str(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        report = explainer.compute_stability(X, n_bootstraps=5)
        text = str(report)
        assert "Stability Report" in text
        assert "Jaccard" in text


class TestConfidenceIntervals:
    def test_ci_fidelity(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=4)
        cis = explainer.compute_confidence_intervals(
            result, X, y=y, n_bootstraps=100,
        )

        assert "fidelity" in cis
        assert "accuracy" in cis
        fi = cis["fidelity"]
        assert fi.lower <= fi.point_estimate <= fi.upper
        ai = cis["accuracy"]
        assert ai.lower <= ai.point_estimate <= ai.upper

    def test_ci_without_y(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, max_depth=4)
        cis = explainer.compute_confidence_intervals(result, X, n_bootstraps=50)

        assert "fidelity" in cis
        assert "accuracy" not in cis


class TestNormalizedComplexity:
    def test_report_has_normalized_metrics(self, iris_data):
        X, y, feat, cls = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(rf, feature_names=feat, class_names=cls)
        result = explainer.extract_rules(X, y=y, max_depth=4)
        report = result.report

        assert report.avg_conditions_per_feature is not None
        assert report.interaction_strength is not None
        assert report.avg_conditions_per_feature >= 0
        assert 0.0 <= report.interaction_strength <= 1.0
