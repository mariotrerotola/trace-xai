"""Test end-to-end della pipeline completa di TRACE-XAI.

Combina più moduli in sequenza su scenari realistici:
  extract_rules → pruning → counterfactual → MDL → stability → metrics
"""

import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from trace_xai import (
    Explainer,
    ExplanationResult,
    PruningConfig,
    export_html,
)
from trace_xai.mdl_selection import MDLSelectionReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def iris_pipeline():
    """Iris: 4 feature, 3 classi, RandomForest."""
    iris = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42,
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    explainer = Explainer(
        model,
        feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
    )
    return explainer, X_tr, X_te, y_tr, y_te


@pytest.fixture()
def wine_pipeline():
    """Wine: 13 feature, 3 classi, GradientBoosting."""
    wine = load_wine()
    X_tr, X_te, y_tr, y_te = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42,
    )
    model = GradientBoostingClassifier(
        n_estimators=30, max_depth=3, random_state=42,
    ).fit(X_tr, y_tr)
    explainer = Explainer(
        model,
        feature_names=list(wine.feature_names),
        class_names=[f"class_{i}" for i in range(3)],
    )
    return explainer, X_tr, X_te, y_tr, y_te


@pytest.fixture()
def cancer_pipeline():
    """Breast Cancer: 30 feature, 2 classi, MLP."""
    cancer = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        cancer.data, cancer.target, test_size=0.3, random_state=42,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=500, random_state=42,
    ).fit(X_tr, y_tr)
    explainer = Explainer(
        model,
        feature_names=list(cancer.feature_names),
        class_names=list(cancer.target_names),
    )
    return explainer, X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Test: pipeline completa in un singolo extract_rules()
# ---------------------------------------------------------------------------

class TestSingleCallPipeline:
    """Testa la combinazione di più feature in una singola chiamata a extract_rules."""

    def test_pruning_plus_mdl(self, iris_pipeline):
        """Pruning + MDL selection nella stessa estrazione."""
        explainer, X_tr, X_te, y_tr, y_te = iris_pipeline
        result = explainer.extract_rules(
            X_tr,
            y=y_tr,
            max_depth=6,
            pruning=PruningConfig(min_confidence=0.6, min_samples=3),
            mdl_selection="forward",
        )
        assert isinstance(result, ExplanationResult)
        assert result.pruning_report is not None
        assert result.mdl_report is not None
        assert result.rules.num_rules <= result.mdl_report.n_rules_original

    def test_counterfactual_plus_mdl(self, iris_pipeline):
        """Counterfactual filtering + MDL in una sola chiamata."""
        explainer, X_tr, X_te, y_tr, y_te = iris_pipeline
        result = explainer.extract_rules(
            X_tr,
            y=y_tr,
            max_depth=5,
            counterfactual_validity_threshold=0.3,
            counterfactual_n_probes=10,
            mdl_selection="forward",
        )
        assert result.counterfactual_report is not None
        assert result.mdl_report is not None

    def test_holdout_validation(self, iris_pipeline):
        """Estrazione con hold-out validation separato."""
        explainer, X_tr, X_te, y_tr, y_te = iris_pipeline
        result = explainer.extract_rules(
            X_tr,
            y=y_tr,
            X_val=X_te,
            y_val=y_te,
            max_depth=4,
        )
        assert result.train_report is not None
        assert result.report is not None
        # Il report principale è sul validation set
        assert result.report.evaluation_type == "hold_out"

    def test_validation_split(self, iris_pipeline):
        """Estrazione con validation split interno."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        result = explainer.extract_rules(
            X_tr,
            y=y_tr,
            validation_split=0.2,
            max_depth=4,
        )
        assert result.train_report is not None

    def test_augmentation_plus_pruning(self, iris_pipeline):
        """Data augmentation + pruning."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        result = explainer.extract_rules(
            X_tr,
            y=y_tr,
            max_depth=5,
            augmentation="perturbation",
            pruning=PruningConfig(min_confidence=0.7, remove_redundant=True),
        )
        assert result.pruning_report is not None
        # L'augmentation non produce un report, ma il surrogate è addestrato su dati aumentati
        assert result.surrogate is not None

    def test_full_pipeline_single_call(self, iris_pipeline):
        """Combina TUTTE le feature disponibili in extract_rules."""
        explainer, X_tr, X_te, y_tr, y_te = iris_pipeline
        result = explainer.extract_rules(
            X_tr,
            y=y_tr,
            max_depth=4,
            X_val=X_te,
            y_val=y_te,
            augmentation="perturbation",
            pruning=PruningConfig(min_confidence=0.3, min_samples=2),
            counterfactual_validity_threshold=0.1,
            counterfactual_n_probes=10,
            mdl_selection="forward",
        )
        assert result.train_report is not None
        assert result.pruning_report is not None
        assert result.counterfactual_report is not None
        assert result.mdl_report is not None
        # Con tutti i filtri attivi, potremmo avere poche regole ma almeno il report c'è
        assert result.mdl_report.n_rules_original > 0


# ---------------------------------------------------------------------------
# Test: pipeline multi-step con chiamate separate
# ---------------------------------------------------------------------------

class TestMultiStepPipeline:
    """Testa la pipeline con chiamate separate dopo l'estrazione."""

    def test_extract_then_stability(self, iris_pipeline):
        """Estrai regole, poi calcola stabilità strutturale."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        stability = explainer.compute_structural_stability(X_tr, max_depth=4)
        assert hasattr(stability, "mean_prediction_agreement")
        assert 0.0 <= stability.mean_prediction_agreement <= 1.0

    def test_extract_then_complementary_metrics(self, iris_pipeline):
        """Estrai regole, poi calcola metriche complementari."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        result = explainer.extract_rules(X_tr, y=y_tr, max_depth=4)
        metrics = explainer.compute_complementary_metrics(result, X_tr)
        assert hasattr(metrics, "boundary_agreement")
        assert hasattr(metrics, "effective_complexity")

    def test_extract_then_cv_fidelity(self, iris_pipeline):
        """Estrai regole, poi cross-validate fidelity."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        cv_report = explainer.cross_validate_fidelity(
            X_tr, y=y_tr, n_folds=3, max_depth=4,
        )
        assert cv_report.n_folds == 3
        assert 0.0 <= cv_report.mean_fidelity <= 1.0

    def test_extract_then_confidence_intervals(self, iris_pipeline):
        """Estrai regole, poi calcola intervalli di confidenza bootstrap."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        result = explainer.extract_rules(X_tr, y=y_tr, max_depth=4)
        cis = explainer.compute_confidence_intervals(
            result, X_tr, y=y_tr, n_bootstraps=20,
        )
        assert "fidelity" in cis
        assert cis["fidelity"].lower <= cis["fidelity"].upper

    def test_extract_then_fidelity_bounds(self, iris_pipeline):
        """Estrai regole, poi calcola bound teorici."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        result = explainer.extract_rules(X_tr, y=y_tr, max_depth=4)
        bounds = explainer.compute_fidelity_bounds(result)
        assert hasattr(bounds, "vc_dimension")
        assert bounds.vc_dimension > 0

    def test_full_multi_step_pipeline(self, wine_pipeline):
        """Pipeline completa multi-step su Wine dataset."""
        explainer, X_tr, X_te, y_tr, y_te = wine_pipeline

        # Step 1: estrazione con hold-out e MDL
        result = explainer.extract_rules(
            X_tr,
            y=y_tr,
            X_val=X_te,
            y_val=y_te,
            max_depth=5,
            mdl_selection="forward",
        )
        assert result.mdl_report is not None
        n_rules_after_mdl = result.rules.num_rules

        # Step 2: counterfactual scoring post-hoc
        cf_report = explainer.score_rules_counterfactual(result, X_tr)
        assert cf_report.n_rules_total == n_rules_after_mdl

        # Step 3: metriche complementari
        metrics = explainer.compute_complementary_metrics(result, X_tr)
        assert metrics.effective_complexity >= 0

        # Step 4: stabilità strutturale
        stability = explainer.compute_structural_stability(X_tr, max_depth=5)
        assert 0.0 <= stability.mean_prediction_agreement <= 1.0

        # Step 5: bound teorici
        bounds = explainer.compute_fidelity_bounds(result)
        assert bounds.vc_dimension > 0


# ---------------------------------------------------------------------------
# Test: ensemble extraction
# ---------------------------------------------------------------------------

class TestEnsemblePipeline:
    """Testa la pipeline con estrazione ensemble (regole stabili)."""

    def test_stable_rules_extraction(self, iris_pipeline):
        """Estrai regole stabili con ensemble bagging."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        result = explainer.extract_stable_rules(
            X_tr, y=y_tr,
            n_estimators=10,
            frequency_threshold=0.3,
            max_depth=4,
        )
        assert result.ensemble_report is not None
        assert result.rules.num_rules > 0

    def test_stable_rules_then_mdl(self, iris_pipeline):
        """Estrai regole stabili, poi applica MDL."""
        explainer, X_tr, _, y_tr, _ = iris_pipeline
        result = explainer.extract_stable_rules(
            X_tr, y=y_tr,
            n_estimators=10,
            frequency_threshold=0.3,
            max_depth=4,
        )
        if result.rules.num_rules > 0:
            mdl_report = explainer.select_rules_mdl(result, X_tr)
            assert isinstance(mdl_report, MDLSelectionReport)
            assert mdl_report.n_rules_selected <= mdl_report.n_rules_original


# ---------------------------------------------------------------------------
# Test: multi-dataset per verificare robustezza
# ---------------------------------------------------------------------------

class TestMultiDataset:
    """Verifica che la pipeline funzioni su dataset diversi."""

    def test_wine_high_dimensional(self, wine_pipeline):
        """Wine: 13 feature con GradientBoosting."""
        explainer, X_tr, X_te, y_tr, y_te = wine_pipeline
        result = explainer.extract_rules(
            X_tr, y=y_tr, max_depth=5,
            X_val=X_te, y_val=y_te,
            mdl_selection="forward",
        )
        assert result.rules.num_rules > 0
        assert result.report.fidelity > 0.5

    def test_cancer_30_features_mlp(self, cancer_pipeline):
        """Breast Cancer: 30 feature, 2 classi, MLP black-box."""
        explainer, X_tr, X_te, y_tr, y_te = cancer_pipeline
        result = explainer.extract_rules(
            X_tr, y=y_tr, max_depth=4,
            X_val=X_te, y_val=y_te,
            pruning=PruningConfig(min_confidence=0.7),
            mdl_selection="backward",
        )
        assert result.rules.num_rules > 0
        assert result.pruning_report is not None
        assert result.mdl_report is not None

    def test_fidelity_across_datasets(self, iris_pipeline, wine_pipeline, cancer_pipeline):
        """La fidelity deve essere ragionevole su tutti i dataset."""
        for name, (explainer, X_tr, _, y_tr, _) in [
            ("iris", iris_pipeline),
            ("wine", wine_pipeline),
            ("cancer", cancer_pipeline),
        ]:
            result = explainer.extract_rules(X_tr, y=y_tr, max_depth=5)
            assert result.report.fidelity > 0.7, (
                f"Fidelity troppo bassa su {name}: {result.report.fidelity:.2f}"
            )


# ---------------------------------------------------------------------------
# Test: export HTML
# ---------------------------------------------------------------------------

class TestHTMLExportPipeline:
    """Testa l'export HTML dopo una pipeline completa."""

    def test_export_after_full_pipeline(self, iris_pipeline):
        """L'export HTML funziona dopo pruning + MDL."""
        explainer, X_tr, X_te, y_tr, y_te = iris_pipeline
        result = explainer.extract_rules(
            X_tr, y=y_tr, max_depth=5,
            X_val=X_te, y_val=y_te,
            pruning=PruningConfig(min_confidence=0.5),
            mdl_selection="forward",
        )
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result.to_html(path)
            assert os.path.exists(path)
            content = open(path).read()
            assert "<html" in content.lower()
            assert len(content) > 500
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: stampa output di esempio della pipeline completa
# ---------------------------------------------------------------------------

class TestPipelineOutput:
    def test_print_full_pipeline(self, wine_pipeline, capsys):
        """Stampa l'output di una pipeline completa su Wine."""
        explainer, X_tr, X_te, y_tr, y_te = wine_pipeline

        result = explainer.extract_rules(
            X_tr, y=y_tr,
            X_val=X_te, y_val=y_te,
            max_depth=5,
            pruning=PruningConfig(min_confidence=0.6, min_samples=3),
            counterfactual_validity_threshold=0.3,
            counterfactual_n_probes=10,
            mdl_selection="forward",
        )

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETA SU WINE DATASET")
        print("=" * 70)

        print(f"\n--- Fidelity Report ({result.report.evaluation_type}) ---")
        print(f"  Fidelity: {result.report.fidelity:.2%}")
        if result.report.accuracy is not None:
            print(f"  Accuracy: {result.report.accuracy:.2%}")
        print(f"  Num rules: {result.report.num_rules}")

        if result.train_report:
            print(f"\n--- Train Report ---")
            print(f"  Fidelity: {result.train_report.fidelity:.2%}")

        if result.pruning_report:
            print(f"\n--- Pruning ---")
            print(f"  Originali: {result.pruning_report.original_count}")
            print(f"  Dopo pruning: {result.pruning_report.pruned_count}")

        if result.counterfactual_report:
            print(f"\n--- Counterfactual ---")
            print(f"  Mean validity: {result.counterfactual_report.mean_score:.2f}")
            print(f"  Rules scored: {result.counterfactual_report.n_rules_total}")

        if result.mdl_report:
            print(f"\n--- MDL Selection ---")
            print(f"  Precision bits: {result.mdl_report.precision_bits}")
            print(f"  Rules: {result.mdl_report.n_rules_original} -> {result.mdl_report.n_rules_selected}")
            print(f"  MDL reduction: {result.mdl_report.mdl_reduction:.1f} bits")

        print(f"\n--- Regole finali ({result.rules.num_rules}) ---")
        for i, rule in enumerate(result.rules.rules):
            conds = " AND ".join(
                f"{c.feature} {c.operator} {c.threshold:.2f}"
                for c in rule.conditions
            )
            print(f"  {i + 1}. IF {conds} THEN {rule.prediction}")

        print("=" * 70)

        captured = capsys.readouterr()
        assert len(captured.out) > 0
