"""Tests for regression support."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from trace_xai import Explainer, ExplanationResult, RuleSet
from trace_xai.report import FidelityReport


@pytest.fixture()
def regression_data():
    X, y = make_regression(
        n_samples=200, n_features=5, n_informative=3, random_state=42
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


class TestRegressionExplainer:
    def test_auto_detect_regression(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        assert explainer._task == "regression"

    def test_explicit_task(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat, task="regression")
        assert explainer._task == "regression"

    def test_invalid_task(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        with pytest.raises(ValueError, match="task must be"):
            Explainer(model, feature_names=feat, task="invalid")

    def test_extract_rules(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        result = explainer.extract_rules(X, y=y, max_depth=4)

        assert isinstance(result, ExplanationResult)
        assert isinstance(result.rules, RuleSet)
        assert result.rules.num_rules > 0
        assert result._task == "regression"

    def test_regression_report_metrics(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        result = explainer.extract_rules(X, y=y, max_depth=4)
        report = result.report

        assert report.fidelity_r2 is not None
        assert report.fidelity_mse is not None
        assert report.accuracy_r2 is not None
        assert report.accuracy_mse is not None
        assert report.fidelity_mse >= 0.0

    def test_regression_rule_format(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        result = explainer.extract_rules(X, max_depth=3)

        for rule in result.rules.rules:
            text = str(rule)
            assert "THEN value =" in text
            assert rule.prediction_value is not None

    def test_regression_report_str(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        result = explainer.extract_rules(X, y=y, max_depth=3)
        text = str(result.report)

        assert "Fidelity R" in text
        assert "Fidelity MSE" in text

    def test_regression_cv_fidelity(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        cv_report = explainer.cross_validate_fidelity(X, y=y, n_folds=3)

        assert cv_report.n_folds == 3
        assert len(cv_report.fold_reports) == 3
        assert cv_report.std_fidelity >= 0.0

    def test_regression_stability(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        stability = explainer.compute_stability(X, n_bootstraps=5)

        assert 0.0 <= stability.mean_jaccard <= 1.0
        assert stability.n_bootstraps == 5

    def test_regression_holdout(self, regression_data):
        X, y, feat = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        explainer = Explainer(model, feature_names=feat)
        result = explainer.extract_rules(
            X[:150], y=y[:150], X_val=X[150:], y_val=y[150:],
        )

        assert result.report.evaluation_type == "hold_out"
        assert result.train_report is not None
        assert result.train_report.evaluation_type == "in_sample"
