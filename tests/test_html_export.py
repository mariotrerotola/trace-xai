"""Tests for HTML export."""

import os
import tempfile

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer


@pytest.fixture()
def iris_result():
    iris = load_iris()
    X, y = iris.data, iris.target
    feat = list(iris.feature_names)
    cls = list(iris.target_names)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    explainer = Explainer(model, feature_names=feat, class_names=cls)
    return explainer.extract_rules(X, y=y, max_depth=3)


class TestHTMLExport:
    def test_creates_file(self, iris_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            iris_result.to_html(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_html_structure(self, iris_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            iris_result.to_html(path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "<table" in html
            assert "Fidelity Report" in html
            assert "filter-btn" in html

    def test_contains_rules(self, iris_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            iris_result.to_html(path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            # Should contain feature names from iris
            assert "petal" in html.lower() or "sepal" in html.lower()

    def test_has_search_input(self, iris_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            iris_result.to_html(path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert 'id="search"' in html
