"""Tests for theoretical_bounds.py - PAC/VC/Rademacher fidelity bounds."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer
from trace_xai.theoretical_bounds import (
    FidelityBound,
    compute_fidelity_bounds,
    estimation_error_pac,
    estimation_error_rademacher,
    min_depth_for_regions,
    optimal_depth_bound,
    relu_network_regions,
    sample_complexity,
    vc_dimension_decision_tree,
)


class TestVCDimension:
    def test_positive_result(self):
        v = vc_dimension_decision_tree(depth=3, n_features=10)
        assert v > 0

    def test_increases_with_depth(self):
        v3 = vc_dimension_decision_tree(3, 10)
        v5 = vc_dimension_decision_tree(5, 10)
        assert v5 > v3

    def test_increases_with_features(self):
        v10 = vc_dimension_decision_tree(3, 10)
        v100 = vc_dimension_decision_tree(3, 100)
        assert v100 > v10

    def test_zero_depth(self):
        assert vc_dimension_decision_tree(0, 10) == 0

    def test_zero_features(self):
        assert vc_dimension_decision_tree(3, 0) == 0

    def test_depth_1(self):
        # Depth 1: 1 internal node, V = ceil(1 * (1 + log2(d)))
        v = vc_dimension_decision_tree(1, 4)
        assert v > 0


class TestEstimationErrorPAC:
    def test_decreases_with_samples(self):
        e100 = estimation_error_pac(50, 100)
        e1000 = estimation_error_pac(50, 1000)
        assert e1000 < e100

    def test_increases_with_vc_dim(self):
        e10 = estimation_error_pac(10, 500)
        e100 = estimation_error_pac(100, 500)
        assert e100 > e10

    def test_zero_samples(self):
        assert estimation_error_pac(10, 0) == 1.0

    def test_bounded_0_1(self):
        e = estimation_error_pac(50, 200, delta=0.05)
        assert 0.0 <= e <= 1.0


class TestEstimationErrorRademacher:
    def test_decreases_with_samples(self):
        e100 = estimation_error_rademacher(3, 10, 100)
        e1000 = estimation_error_rademacher(3, 10, 1000)
        assert e1000 < e100

    def test_increases_with_depth(self):
        e3 = estimation_error_rademacher(3, 10, 500)
        e8 = estimation_error_rademacher(8, 10, 500)
        assert e8 > e3

    def test_zero_samples(self):
        assert estimation_error_rademacher(3, 10, 0) == 1.0


class TestSampleComplexity:
    def test_positive(self):
        n = sample_complexity(50, epsilon=0.05, delta=0.05)
        assert n > 0

    def test_increases_with_vc_dim(self):
        n10 = sample_complexity(10, 0.05, 0.05)
        n100 = sample_complexity(100, 0.05, 0.05)
        assert n100 > n10

    def test_increases_with_tighter_epsilon(self):
        n_loose = sample_complexity(50, epsilon=0.1, delta=0.05)
        n_tight = sample_complexity(50, epsilon=0.01, delta=0.05)
        assert n_tight > n_loose


class TestReLUNetworkRegions:
    def test_single_layer(self):
        r = relu_network_regions(1, 10, 3)
        assert r > 1

    def test_increases_with_layers(self):
        r1 = relu_network_regions(1, 10, 3)
        r3 = relu_network_regions(3, 10, 3)
        assert r3 > r1

    def test_increases_with_width(self):
        r10 = relu_network_regions(2, 10, 3)
        r50 = relu_network_regions(2, 50, 3)
        assert r50 > r10

    def test_zero_layers(self):
        assert relu_network_regions(0, 10, 3) == 1

    def test_caps_overflow(self):
        r = relu_network_regions(100, 1000, 5)
        assert r <= 2**63

    def test_width_less_than_input(self):
        r = relu_network_regions(2, 3, 10)
        assert r >= 1


class TestMinDepthForRegions:
    def test_power_of_two(self):
        assert min_depth_for_regions(8) == 3

    def test_non_power_of_two(self):
        assert min_depth_for_regions(5) == 3  # ceil(log2(5)) = 3

    def test_single_region(self):
        assert min_depth_for_regions(1) == 0

    def test_two_regions(self):
        assert min_depth_for_regions(2) == 1


class TestOptimalDepthBound:
    def test_positive(self):
        d = optimal_depth_bound(1000, 10)
        assert d >= 1

    def test_increases_with_samples(self):
        d100 = optimal_depth_bound(100, 10)
        d10000 = optimal_depth_bound(10000, 10)
        assert d10000 >= d100

    def test_edge_cases(self):
        assert optimal_depth_bound(1, 10) == 1


class TestComputeFidelityBounds:
    def test_returns_fidelity_bound(self):
        bound = compute_fidelity_bounds(
            depth=3, n_features=10, n_samples=500,
            empirical_infidelity=0.05, delta=0.05,
        )
        assert isinstance(bound, FidelityBound)

    def test_fields(self):
        bound = compute_fidelity_bounds(
            depth=3, n_features=10, n_samples=500,
        )
        assert bound.vc_dimension > 0
        assert 0.0 <= bound.estimation_error_pac <= 1.0
        assert 0.0 <= bound.estimation_error_rademacher <= 1.0
        assert 0.0 <= bound.min_fidelity_pac <= 1.0
        assert 0.0 <= bound.min_fidelity_rademacher <= 1.0
        assert bound.max_decision_regions == 8
        assert bound.confidence == 0.95

    def test_more_samples_tighter_bounds(self):
        b100 = compute_fidelity_bounds(3, 10, 100)
        b10000 = compute_fidelity_bounds(3, 10, 10000)
        assert b10000.estimation_error_pac < b100.estimation_error_pac
        assert b10000.min_fidelity_pac > b100.min_fidelity_pac

    def test_str_output(self):
        bound = compute_fidelity_bounds(3, 10, 500)
        text = str(bound)
        assert "Theoretical Fidelity Bounds" in text
        assert "VC-dimension" in text
        assert "PAC" in text

    def test_high_empirical_infidelity(self):
        bound = compute_fidelity_bounds(3, 10, 500, empirical_infidelity=0.9)
        assert bound.min_fidelity_pac >= 0.0  # clamped


class TestExplainerIntegration:
    def test_compute_fidelity_bounds(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(
            rf, feature_names=list(iris.feature_names),
            class_names=list(iris.target_names),
        )
        result = explainer.extract_rules(X, y=y, max_depth=3)
        bound = explainer.compute_fidelity_bounds(result)
        assert isinstance(bound, FidelityBound)
        assert bound.depth == result.report.surrogate_depth
        assert bound.n_samples == result.report.num_samples
        assert bound.min_fidelity_pac >= 0.0
