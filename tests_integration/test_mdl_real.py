"""Test realistici end-to-end per il modulo MDL Selection.

Questi test verificano il funzionamento del modulo MDL su un caso d'uso
reale: dataset Iris, modello RandomForest, train/test split.
"""

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from trace_xai import Explainer
from trace_xai.mdl_selection import (
    MDLSelectionReport,
    select_rules_mdl,
)


@pytest.fixture()
def real_setup():
    """Setup realistico con train/test split."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42,
    )
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    explainer = Explainer(
        rf,
        feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
    )
    return explainer, X_train, X_test, y_train, y_test


@pytest.fixture()
def rules_deep(real_setup):
    """Estrae regole con albero profondo per generare molte regole."""
    explainer, X_train, X_test, y_train, y_test = real_setup
    result = explainer.extract_rules(X_train, y=y_train, max_depth=6)
    return result, explainer, X_train, X_test, y_train, y_test


class TestMDLForwardSelection:
    def test_reduces_rules(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward",
        )
        assert report.n_rules_selected <= report.n_rules_original

    def test_reduces_total_mdl(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward",
        )
        assert report.total_mdl_after <= report.total_mdl_before + 1e-9
        assert report.mdl_reduction >= -1e-9


class TestMDLBackwardElimination:
    def test_reduces_rules(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="backward",
        )
        assert report.n_rules_selected <= report.n_rules_original

    def test_selected_ruleset_is_valid(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="backward",
        )
        assert report.selected_ruleset.num_rules == report.n_rules_selected
        assert report.selected_ruleset.feature_names == result.rules.feature_names


class TestMDLFidelity:
    def test_selected_rules_maintain_fidelity(self, rules_deep):
        """Le regole selezionate devono coprire i campioni con buona accuratezza."""
        result, explainer, X_train, X_test, y_train, y_test = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward",
        )
        # Verifica che le regole selezionate abbiano error_rate ragionevole
        selected_indices = [
            i for i, rs in enumerate(report.rule_scores)
            if rs.rule in report.selected_ruleset.rules
        ]
        for i in selected_indices:
            score = report.rule_scores[i]
            if score.coverage > 0:
                assert score.error_rate < 0.5, (
                    f"Regola {i} ha error_rate {score.error_rate:.2f}, "
                    f"troppo alto per una selezione MDL"
                )


class TestMDLScoresDecomposition:
    def test_model_plus_data_equals_total(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="score_only",
        )
        for score in report.rule_scores:
            assert score.total_mdl == pytest.approx(
                score.model_cost + score.data_cost,
            )

    def test_all_costs_nonnegative(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="score_only",
        )
        for score in report.rule_scores:
            assert score.model_cost >= 0.0
            assert score.data_cost >= 0.0
            assert score.total_mdl >= 0.0
            assert 0.0 <= score.error_rate <= 1.0
            assert score.coverage >= 0


class TestMDLForwardVsBackward:
    def test_both_methods_produce_valid_results(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        fwd = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward",
        )
        bwd = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="backward",
        )
        assert isinstance(fwd, MDLSelectionReport)
        assert isinstance(bwd, MDLSelectionReport)
        assert fwd.n_rules_selected <= fwd.n_rules_original
        assert bwd.n_rules_selected <= bwd.n_rules_original
        # Entrambi partono dallo stesso insieme di regole
        assert fwd.n_rules_original == bwd.n_rules_original
        assert fwd.total_mdl_before == pytest.approx(bwd.total_mdl_before)


class TestMDLIntegratedPipeline:
    def test_extract_rules_with_mdl_selection(self, real_setup):
        """Flusso completo: extract_rules con mdl_selection integrato."""
        explainer, X_train, X_test, y_train, y_test = real_setup
        result = explainer.extract_rules(
            X_train, y=y_train, max_depth=6, mdl_selection="forward",
        )
        assert result.mdl_report is not None
        assert isinstance(result.mdl_report, MDLSelectionReport)
        assert result.rules.num_rules == result.mdl_report.n_rules_selected
        assert result.mdl_report.selection_method == "forward"

    def test_explainer_select_rules_mdl_method(self, real_setup):
        """Test del metodo select_rules_mdl sull'Explainer."""
        explainer, X_train, X_test, y_train, y_test = real_setup
        result = explainer.extract_rules(X_train, y=y_train, max_depth=6)
        report = explainer.select_rules_mdl(result, X_train)
        assert isinstance(report, MDLSelectionReport)
        assert report.n_rules_selected <= report.n_rules_original


class TestMDLReportOutput:
    def test_report_is_printable(self, rules_deep):
        result, explainer, X_train, *_ = rules_deep
        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward",
        )
        text = str(report)
        assert "MDL Selection Report" in text
        assert "Rules:" in text
        assert "forward" in text
        assert "bits" in text

    def test_print_example_rules(self, rules_deep, capsys):
        """Stampa le regole originali, il report MDL e le regole selezionate."""
        result, explainer, X_train, *_ = rules_deep

        print("\n" + "=" * 60)
        print("REGOLE ORIGINALI (prima della selezione MDL)")
        print("=" * 60)
        for i, rule in enumerate(result.rules.rules):
            conditions = " AND ".join(
                f"{c.feature} {c.operator} {c.threshold:.2f}"
                for c in rule.conditions
            )
            print(
                f"  Rule {i + 1}: IF {conditions} "
                f"THEN {rule.prediction} "
                f"(confidence={rule.confidence:.2f}, samples={rule.samples})"
            )
        print(f"\nTotale regole originali: {result.rules.num_rules}")

        report = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward",
        )

        print("\n" + "=" * 60)
        print("MDL REPORT")
        print("=" * 60)
        print(report)

        print("\n" + "=" * 60)
        print("SCORES PER REGOLA")
        print("=" * 60)
        for score in report.rule_scores:
            conditions = " AND ".join(
                f"{c.feature} {c.operator} {c.threshold:.2f}"
                for c in score.rule.conditions
            )
            selected = "✓" if score.rule in report.selected_ruleset.rules else "✗"
            print(
                f"  [{selected}] Rule {score.rule_index + 1}: "
                f"model_cost={score.model_cost:.1f} "
                f"data_cost={score.data_cost:.1f} "
                f"total_mdl={score.total_mdl:.1f} "
                f"coverage={score.coverage} "
                f"error_rate={score.error_rate:.2%}"
            )
            print(f"       IF {conditions} THEN {score.rule.prediction}")

        print("\n" + "=" * 60)
        print("REGOLE SELEZIONATE (dopo MDL forward)")
        print("=" * 60)
        for i, rule in enumerate(report.selected_ruleset.rules):
            conditions = " AND ".join(
                f"{c.feature} {c.operator} {c.threshold:.2f}"
                for c in rule.conditions
            )
            print(
                f"  Rule {i + 1}: IF {conditions} "
                f"THEN {rule.prediction} "
                f"(confidence={rule.confidence:.2f}, samples={rule.samples})"
            )
        print(f"\nTotale regole selezionate: {report.n_rules_selected}")
        print(f"Riduzione MDL: {report.mdl_reduction:.2f} bits")
        print("=" * 60)

        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestMDLAutoPrecision:
    def test_auto_precision_selects_more_rules(self, rules_deep):
        """Con precision_bits='auto', vengono selezionate più regole che con 16."""
        result, explainer, X_train, *_ = rules_deep
        report_fixed = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward", precision_bits=16,
        )
        report_auto = select_rules_mdl(
            result.rules, explainer.model, X_train,
            n_classes=3, method="forward", precision_bits="auto",
        )
        assert report_auto.n_rules_selected >= report_fixed.n_rules_selected
        assert report_auto.precision_bits < 16
