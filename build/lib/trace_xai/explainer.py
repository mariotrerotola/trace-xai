"""Core explainer: wraps any black-box model and extracts interpretable rules."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
from .visualization import export_dot, plot_surrogate_tree


class ExplanationResult:
    """Container returned by :meth:`Explainer.extract_rules`.

    Attributes
    ----------
    rules : RuleSet
        The extracted IF-THEN rules.
    report : FidelityReport
        Quantitative fidelity metrics.
    surrogate
        The fitted surrogate tree (for advanced usage).
    train_report : FidelityReport or None
        In-sample fidelity report (only when hold-out evaluation is used).
    """

    def __init__(
        self,
        rules: RuleSet,
        report: FidelityReport,
        surrogate,
        feature_names: tuple[str, ...],
        class_names: Optional[tuple[str, ...]] = None,
        *,
        train_report: Optional[FidelityReport] = None,
        task: str = "classification",
        pruned_rules: Optional[RuleSet] = None,
        pruning_report=None,
        monotonicity_report=None,
        ensemble_report=None,
        stable_rules=None,
    ) -> None:
        self.rules = rules
        self.report = report
        self.surrogate = surrogate
        self._feature_names = feature_names
        self._class_names = class_names or ()
        self.train_report = train_report
        self._task = task
        self.pruned_rules = pruned_rules
        self.pruning_report = pruning_report
        self.monotonicity_report = monotonicity_report
        self.ensemble_report = ensemble_report
        self.stable_rules = stable_rules

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def plot(self, *, save_path: Optional[str] = None, **kwargs) -> None:
        """Render the surrogate tree (delegates to :func:`plot_surrogate_tree`)."""
        plot_surrogate_tree(
            self.surrogate,
            self._feature_names,
            self._class_names,
            save_path=save_path,
            **kwargs,
        )

    def to_dot(self) -> str:
        """Export the surrogate tree as a Graphviz DOT string."""
        return export_dot(self.surrogate, self._feature_names, self._class_names)

    def to_html(self, output_path: str) -> None:
        """Export an interactive HTML report."""
        from .html_export import export_html

        export_html(self, output_path)

    def __str__(self) -> str:
        parts = [str(self.rules), "", str(self.report)]
        if self.train_report is not None:
            parts += ["", "--- Train (in-sample) report ---", str(self.train_report)]
        if self.pruning_report is not None:
            parts += ["", f"--- Pruning Report ---",
                       f"  Original rules: {self.pruning_report.original_count}",
                       f"  Pruned rules: {self.pruning_report.pruned_count}",
                       f"  Removed (low confidence): {self.pruning_report.removed_low_confidence}",
                       f"  Removed (low samples): {self.pruning_report.removed_low_samples}",
                       f"  Conditions simplified: {self.pruning_report.conditions_simplified}"]
        if self.monotonicity_report is not None:
            parts += ["", str(self.monotonicity_report)]
        if self.ensemble_report is not None:
            parts += ["", str(self.ensemble_report)]
        return "\n".join(parts)


class Explainer:
    """Model-agnostic rule extractor.

    Parameters
    ----------
    model : object
        Any model exposing a ``predict(X)`` method.
    feature_names : sequence of str
        Human-readable feature names (length must match ``X.shape[1]``).
    class_names : sequence of str or None
        Human-readable class names. Required for classification, omit for
        regression (or set ``task="regression"``).
    task : str or None
        ``"classification"`` or ``"regression"``.  If *None*, auto-detected:
        classification when *class_names* is provided, regression otherwise.
    """

    def __init__(
        self,
        model: object,
        feature_names: Sequence[str],
        class_names: Optional[Sequence[str]] = None,
        *,
        task: Optional[str] = None,
    ) -> None:
        if not hasattr(model, "predict") or not callable(model.predict):
            raise TypeError(
                f"The model must expose a callable .predict() method, "
                f"got {type(model).__name__!r}."
            )
        self.model = model
        self.feature_names = tuple(feature_names)

        # Auto-detect task
        if task is not None:
            if task not in ("classification", "regression"):
                raise ValueError(
                    f"task must be 'classification' or 'regression', got {task!r}"
                )
            self._task = task
        elif class_names is not None:
            self._task = "classification"
        else:
            self._task = "regression"

        self.class_names: Optional[tuple[str, ...]]
        if class_names is not None:
            self.class_names = tuple(class_names)
        else:
            self.class_names = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_rules(
        self,
        X: ArrayLike,
        *,
        y: Optional[ArrayLike] = None,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        ccp_alpha: float = 0.0,
        monotonic_constraints: Optional[Dict[str, int]] = None,
        pruning=None,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        validation_split: Optional[float] = None,
        surrogate_type: str = "decision_tree",
        augmentation: Optional[str] = None,
        augmentation_kwargs: Optional[dict] = None,
        preset: Optional[str] = None,
    ) -> ExplanationResult:
        """Extract interpretable rules from the black-box model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training / explanation data.
        y : array-like of shape (n_samples,), optional
            True labels.  Used only for accuracy metrics in the report.
        max_depth : int, default 5
            Maximum depth of the surrogate tree.
        min_samples_leaf : int, default 5
            Minimum samples per leaf in the surrogate tree.
        ccp_alpha : float, default 0.0
            Cost-complexity pruning parameter for the sklearn tree.
        monotonic_constraints : dict mapping feature name to {+1, -1, 0}, optional
            Enforce monotonic relationships during surrogate fitting.
            Requires scikit-learn >= 1.4.
        pruning : PruningConfig, optional
            Post-hoc rule pruning configuration.
        X_val : array-like, optional
            Separate validation set for hold-out fidelity evaluation.
        y_val : array-like, optional
            True labels for the validation set.
        validation_split : float, optional
            If given (0 < value < 1), split *X* internally for hold-out
            evaluation.  Mutually exclusive with *X_val*.
        surrogate_type : str, default "decision_tree"
            Type of surrogate: ``"decision_tree"`` or ``"oblique_tree"``.
        augmentation : str, optional
            Data augmentation strategy: ``"perturbation"``, ``"boundary"``,
            ``"sparse"``, or ``"combined"``. If None, no augmentation.
        augmentation_kwargs : dict, optional
            Extra keyword arguments for the augmentation function.
        preset : str, optional
            Hyperparameter preset name (``"interpretable"``, ``"balanced"``,
            ``"faithful"``). Overrides max_depth, min_samples_leaf, ccp_alpha.

        Returns
        -------
        ExplanationResult
        """
        _supported_surrogates = ("decision_tree", "oblique_tree")
        if surrogate_type not in _supported_surrogates:
            raise ValueError(
                f"Supported surrogates: {_supported_surrogates}, "
                f"got {surrogate_type!r}."
            )

        # Apply preset if given
        if preset is not None:
            from .hyperparams import get_preset
            p = get_preset(preset)
            max_depth = p.max_depth
            min_samples_leaf = p.min_samples_leaf
            ccp_alpha = p.ccp_alpha

        if X_val is not None and validation_split is not None:
            raise ValueError(
                "X_val and validation_split are mutually exclusive."
            )

        X = np.asarray(X)
        y_true = np.asarray(y) if y is not None else None

        # Optional internal split
        X_train, y_train_true = X, y_true
        X_eval: Optional[np.ndarray] = None
        y_eval_true: Optional[np.ndarray] = None
        evaluation_type = "in_sample"

        if X_val is not None:
            X_eval = np.asarray(X_val)
            y_eval_true = np.asarray(y_val) if y_val is not None else None
            evaluation_type = "hold_out"
        elif validation_split is not None:
            from sklearn.model_selection import train_test_split

            split_kwargs: dict = {"test_size": validation_split, "random_state": 42}
            if y_true is not None and self._task == "classification":
                split_kwargs["stratify"] = y_true
            if y_true is not None:
                X_train, X_eval, y_train_true, y_eval_true = train_test_split(
                    X, y_true, **split_kwargs
                )
            else:
                X_train, X_eval = train_test_split(X, **split_kwargs)
                y_train_true = None
                y_eval_true = None
            evaluation_type = "validation_split"

        # 1. Black-box predictions (on training portion)
        y_bb = np.asarray(self.model.predict(X_train))

        # 1b. Data augmentation (query synthesis)
        if augmentation is not None:
            from .augmentation import augment_data
            aug_kw = augmentation_kwargs or {}
            # First pass: train a preliminary surrogate for boundary/sparse strategies
            if augmentation in ("boundary", "sparse", "combined"):
                is_reg = self._task == "regression"
                prelim = self._build_surrogate(
                    is_reg, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                )
                prelim.fit(X_train, y_bb)
                X_train, y_bb = augment_data(
                    X_train, self.model, prelim,
                    strategy=augmentation, **aug_kw,
                )
            else:
                X_train, y_bb = augment_data(
                    X_train, self.model, strategy=augmentation, **aug_kw,
                )
            # Synthetic samples have no true labels
            y_train_true = None

        # 2. Train surrogate
        is_regression = self._task == "regression"
        monotonic_cst = None
        if monotonic_constraints is not None:
            from .monotonicity import check_sklearn_monotonic_support, constraints_to_array
            if not check_sklearn_monotonic_support():
                raise RuntimeError(
                    "Monotonic constraints require scikit-learn >= 1.4. "
                    "Please upgrade: pip install 'scikit-learn>=1.4'"
                )
            monotonic_cst = constraints_to_array(monotonic_constraints, self.feature_names)

        if surrogate_type == "oblique_tree":
            from .surrogates.oblique_tree import ObliqueTreeSurrogate
            surrogate = ObliqueTreeSurrogate(
                task=self._task,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
        else:
            surrogate = self._build_surrogate(
                is_regression,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                monotonic_cst=monotonic_cst,
            )
        surrogate.fit(X_train, y_bb)

        # 3. Extract rules via DFS
        if surrogate_type == "oblique_tree":
            # Use augmented feature names for oblique tree
            aug_names = surrogate.get_augmented_feature_names(self.feature_names)
            rules = self._extract_rules_from_tree(
                surrogate, is_regression=is_regression,
                feature_names_override=aug_names,
            )
        else:
            rules = self._extract_rules_from_tree(surrogate, is_regression=is_regression)
        ruleset = RuleSet(
            rules=tuple(rules),
            feature_names=self.feature_names,
            class_names=self.class_names or (),
        )

        # 3b. Monotonicity validation
        mono_report = None
        if monotonic_constraints is not None:
            from .monotonicity import validate_monotonicity
            mono_report = validate_monotonicity(ruleset, monotonic_constraints)

        # 3c. Post-hoc pruning
        pruned_rules = None
        pruning_report = None
        if pruning is not None:
            from .pruning import prune_ruleset
            pruned_rules, pruning_report = prune_ruleset(ruleset, pruning)

        # Shared kwargs for report builders
        report_kwargs = dict(
            avg_conditions_per_feature=ruleset.avg_conditions_per_feature,
            interaction_strength=ruleset.interaction_strength,
        )

        # 4. Build reports
        train_report: Optional[FidelityReport] = None

        if X_eval is not None:
            # Hold-out evaluation
            y_bb_eval = np.asarray(self.model.predict(X_eval))
            report = self._build_report(
                surrogate, X_eval, y_bb_eval, y_eval_true, ruleset,
                evaluation_type=evaluation_type, **report_kwargs,
            )
            train_report = self._build_report(
                surrogate, X_train, y_bb, y_train_true, ruleset,
                evaluation_type="in_sample", **report_kwargs,
            )
        else:
            report = self._build_report(
                surrogate, X_train, y_bb, y_train_true, ruleset,
                evaluation_type="in_sample", **report_kwargs,
            )

        return ExplanationResult(
            rules=ruleset,
            report=report,
            surrogate=surrogate,
            feature_names=self.feature_names,
            class_names=self.class_names or (),
            train_report=train_report,
            task=self._task,
            pruned_rules=pruned_rules,
            pruning_report=pruning_report,
            monotonicity_report=mono_report,
        )

    def cross_validate_fidelity(
        self,
        X: ArrayLike,
        *,
        y: Optional[ArrayLike] = None,
        n_folds: int = 5,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> CVFidelityReport:
        """Cross-validated fidelity estimation.

        For each fold the surrogate is trained on the training split and
        evaluated on the held-out split.
        """
        X = np.asarray(X)
        y_true = np.asarray(y) if y is not None else None

        if self._task == "classification" and y_true is not None:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            split_iter = cv.split(X, y_true)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            split_iter = cv.split(X)

        fold_reports: list[FidelityReport] = []

        for train_idx, val_idx in split_iter:
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr = y_true[train_idx] if y_true is not None else None
            y_va = y_true[val_idx] if y_true is not None else None

            y_bb_tr = np.asarray(self.model.predict(X_tr))

            is_regression = self._task == "regression"
            surr = self._build_surrogate(
                is_regression,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            surr.fit(X_tr, y_bb_tr)

            rules = self._extract_rules_from_tree(surr, is_regression=is_regression)
            ruleset = RuleSet(
                rules=tuple(rules),
                feature_names=self.feature_names,
                class_names=self.class_names or (),
            )

            y_bb_va = np.asarray(self.model.predict(X_va))
            report = self._build_report(
                surr, X_va, y_bb_va, y_va, ruleset,
                evaluation_type="cross_validation",
                avg_conditions_per_feature=ruleset.avg_conditions_per_feature,
                interaction_strength=ruleset.interaction_strength,
            )
            fold_reports.append(report)

        fidelities = np.array([r.fidelity for r in fold_reports])
        accuracies = [r.accuracy for r in fold_reports]
        has_accuracy = all(a is not None for a in accuracies)

        return CVFidelityReport(
            mean_fidelity=float(fidelities.mean()),
            std_fidelity=float(fidelities.std()),
            mean_accuracy=float(np.mean(accuracies)) if has_accuracy else None,
            std_accuracy=float(np.std(accuracies)) if has_accuracy else None,
            fold_reports=fold_reports,
            n_folds=n_folds,
        )

    def compute_stability(
        self,
        X: ArrayLike,
        *,
        n_bootstraps: int = 20,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        tolerance: Optional[float] = None,
    ) -> StabilityReport:
        """Compute rule stability via bootstrap resampling + Jaccard similarity.

        Parameters
        ----------
        tolerance : float, optional
            If given, use fuzzy rule signatures (thresholds rounded to the
            nearest multiple of *tolerance*) for more realistic Jaccard scores.
        """
        X = np.asarray(X)
        rng = np.random.RandomState(random_state)

        signatures: list[frozenset] = []
        is_regression = self._task == "regression"

        for _ in range(n_bootstraps):
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            y_bb_boot = np.asarray(self.model.predict(X_boot))

            surr = self._build_surrogate(
                is_regression,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            surr.fit(X_boot, y_bb_boot)

            rules = self._extract_rules_from_tree(surr, is_regression=is_regression)
            ruleset = RuleSet(
                rules=tuple(rules),
                feature_names=self.feature_names,
                class_names=self.class_names or (),
            )
            if tolerance is not None:
                signatures.append(ruleset.fuzzy_rule_signatures(tolerance))
            else:
                signatures.append(ruleset.rule_signatures())

        # Pairwise Jaccard
        pairwise: list[float] = []
        for a, b in combinations(range(n_bootstraps), 2):
            union = signatures[a] | signatures[b]
            inter = signatures[a] & signatures[b]
            j = len(inter) / len(union) if union else 1.0
            pairwise.append(j)

        arr = np.array(pairwise) if pairwise else np.array([1.0])
        return StabilityReport(
            mean_jaccard=float(arr.mean()),
            std_jaccard=float(arr.std()),
            pairwise_jaccards=pairwise,
            n_bootstraps=n_bootstraps,
        )

    def compute_confidence_intervals(
        self,
        result: ExplanationResult,
        X: ArrayLike,
        *,
        y: Optional[ArrayLike] = None,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> Dict[str, ConfidenceInterval]:
        """Bootstrap confidence intervals for fidelity (and accuracy if *y* given).

        Returns a dict with keys ``"fidelity"`` and optionally ``"accuracy"``.
        """
        from sklearn.metrics import accuracy_score, r2_score

        X = np.asarray(X)
        y_true = np.asarray(y) if y is not None else None
        rng = np.random.RandomState(random_state)

        y_bb = np.asarray(self.model.predict(X))
        y_surr = result.surrogate.predict(X)

        is_regression = self._task == "regression"

        fidelities: list[float] = []
        accuracies: list[float] = []

        for _ in range(n_bootstraps):
            idx = rng.choice(len(X), size=len(X), replace=True)
            if is_regression:
                fid = float(r2_score(y_bb[idx], y_surr[idx]))
            else:
                fid = float(accuracy_score(y_bb[idx], y_surr[idx]))
            fidelities.append(fid)
            if y_true is not None:
                if is_regression:
                    acc = float(r2_score(y_true[idx], y_surr[idx]))
                else:
                    acc = float(accuracy_score(y_true[idx], y_surr[idx]))
                accuracies.append(acc)

        out: Dict[str, ConfidenceInterval] = {}
        out["fidelity"] = compute_bootstrap_ci(
            np.array(fidelities), confidence_level=confidence_level
        )
        if accuracies:
            out["accuracy"] = compute_bootstrap_ci(
                np.array(accuracies), confidence_level=confidence_level
            )
        return out

    def prune_rules(
        self,
        result: ExplanationResult,
        config,
    ) -> ExplanationResult:
        """Apply post-hoc pruning to an existing ExplanationResult.

        Returns a new ExplanationResult with ``pruned_rules`` and
        ``pruning_report`` populated.
        """
        from .pruning import prune_ruleset

        pruned, report = prune_ruleset(result.rules, config)
        return ExplanationResult(
            rules=result.rules,
            report=result.report,
            surrogate=result.surrogate,
            feature_names=result._feature_names,
            class_names=result._class_names or (),
            train_report=result.train_report,
            task=result._task,
            pruned_rules=pruned,
            pruning_report=report,
            monotonicity_report=result.monotonicity_report,
            ensemble_report=result.ensemble_report,
            stable_rules=result.stable_rules,
        )

    def extract_stable_rules(
        self,
        X: ArrayLike,
        *,
        y: Optional[ArrayLike] = None,
        n_estimators: int = 20,
        frequency_threshold: float = 0.5,
        tolerance: float | dict[str, float] = 0.01,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        ccp_alpha: float = 0.0,
        monotonic_constraints: Optional[Dict[str, int]] = None,
        random_state: int = 42,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
    ) -> ExplanationResult:
        """Extract stable rules via ensemble of bootstrap surrogates.

        Trains *n_estimators* surrogate trees on bootstrap samples, then
        retains only rules appearing in at least *frequency_threshold*
        fraction of trees.

        Parameters
        ----------
        n_estimators : int
            Number of bootstrap surrogate trees.
        frequency_threshold : float
            Minimum fraction of trees in which a rule must appear (0-1).
        tolerance : float or dict
            Threshold rounding tolerance for fuzzy matching.
            Can be a single float (global) or a dict ``{feature: tol}``.
        """
        from .ensemble import extract_ensemble_rules

        X = np.asarray(X)
        y_true = np.asarray(y) if y is not None else None

        # Black-box predictions
        y_bb = np.asarray(self.model.predict(X))

        is_regression = self._task == "regression"
        monotonic_cst = None
        if monotonic_constraints is not None:
            from .monotonicity import check_sklearn_monotonic_support, constraints_to_array
            if not check_sklearn_monotonic_support():
                raise RuntimeError(
                    "Monotonic constraints require scikit-learn >= 1.4."
                )
            monotonic_cst = constraints_to_array(monotonic_constraints, self.feature_names)

        rng = np.random.RandomState(random_state)
        bootstrap_rulesets: list[RuleSet] = []

        for _ in range(n_estimators):
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            y_bb_boot = y_bb[idx]

            surr = self._build_surrogate(
                is_regression,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                monotonic_cst=monotonic_cst,
                random_state=random_state,
            )
            surr.fit(X_boot, y_bb_boot)

            rules = self._extract_rules_from_tree(surr, is_regression=is_regression)
            bootstrap_rulesets.append(RuleSet(
                rules=tuple(rules),
                feature_names=self.feature_names,
                class_names=self.class_names or (),
            ))

        stable_rules, ensemble_report = extract_ensemble_rules(
            bootstrap_rulesets,
            frequency_threshold=frequency_threshold,
            tolerance=tolerance,
        )

        # Build a RuleSet from the stable rules for the result
        stable_ruleset = RuleSet(
            rules=tuple(sr.rule for sr in stable_rules),
            feature_names=self.feature_names,
            class_names=self.class_names or (),
        )

        # Build a report using one of the surrogates (the last one)
        # on the full data for metrics
        surr_final = self._build_surrogate(
            is_regression,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )
        surr_final.fit(X, y_bb)

        report_kwargs = dict(
            avg_conditions_per_feature=stable_ruleset.avg_conditions_per_feature,
            interaction_strength=stable_ruleset.interaction_strength,
        )

        X_eval = np.asarray(X_val) if X_val is not None else None
        y_eval_true = np.asarray(y_val) if y_val is not None else None

        if X_eval is not None:
            y_bb_eval = np.asarray(self.model.predict(X_eval))
            report = self._build_report(
                surr_final, X_eval, y_bb_eval, y_eval_true, stable_ruleset,
                evaluation_type="hold_out", **report_kwargs,
            )
        else:
            report = self._build_report(
                surr_final, X, y_bb, y_true, stable_ruleset,
                evaluation_type="in_sample", **report_kwargs,
            )

        return ExplanationResult(
            rules=stable_ruleset,
            report=report,
            surrogate=surr_final,
            feature_names=self.feature_names,
            class_names=self.class_names or (),
            task=self._task,
            ensemble_report=ensemble_report,
            stable_rules=stable_rules,
        )

    def auto_select_depth(
        self,
        X: ArrayLike,
        *,
        y: Optional[ArrayLike] = None,
        target_fidelity: float = 0.85,
        min_depth: int = 2,
        max_depth: int = 10,
        n_folds: int = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        """Automatically select minimum tree depth achieving target fidelity.

        Delegates to :func:`hyperparams.auto_select_depth`.

        Returns
        -------
        AutoDepthResult
        """
        from .hyperparams import auto_select_depth
        return auto_select_depth(
            self, X, y=y, target_fidelity=target_fidelity,
            min_depth=min_depth, max_depth=max_depth,
            n_folds=n_folds, min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def compute_structural_stability(
        self,
        X: ArrayLike,
        *,
        n_bootstraps: int = 20,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        top_k: int = 3,
        random_state: int = 42,
    ):
        """Compute structural stability (coverage overlap, feature rank stability).

        Delegates to :func:`stability.compute_structural_stability`.

        Returns
        -------
        StructuralStabilityReport
        """
        from .stability import compute_structural_stability
        return compute_structural_stability(
            self, X, n_bootstraps=n_bootstraps, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, top_k=top_k,
            random_state=random_state,
        )

    def compute_complementary_metrics(
        self,
        result: ExplanationResult,
        X: ArrayLike,
    ):
        """Compute metrics beyond standard fidelity.

        Delegates to :func:`metrics.compute_complementary_metrics`.

        Returns
        -------
        ComplementaryMetrics
        """
        from .metrics import compute_complementary_metrics
        return compute_complementary_metrics(
            result.surrogate, self.model, X, result.rules,
            class_names=self.class_names or (),
        )

    def compute_fidelity_bounds(
        self,
        result: ExplanationResult,
        *,
        delta: float = 0.05,
    ):
        """Compute theoretical PAC/VC/Rademacher fidelity bounds.

        Delegates to :func:`theoretical_bounds.compute_fidelity_bounds`.

        Parameters
        ----------
        result : ExplanationResult
            Output from :meth:`extract_rules`.
        delta : float
            Failure probability (bounds hold with prob >= 1 - delta).

        Returns
        -------
        FidelityBound
        """
        from .theoretical_bounds import compute_fidelity_bounds
        empirical_infidelity = 1.0 - result.report.fidelity
        return compute_fidelity_bounds(
            depth=result.report.surrogate_depth,
            n_features=len(result._feature_names),
            n_samples=result.report.num_samples,
            empirical_infidelity=empirical_infidelity,
            delta=delta,
        )

    def sensitivity_analysis(
        self,
        X: ArrayLike,
        *,
        y: Optional[ArrayLike] = None,
        depth_range: Sequence[int] = (3, 5, 7),
        min_samples_leaf_range: Sequence[int] = (5, 10, 20),
        n_folds: int = 3,
        random_state: int = 42,
    ):
        """Grid search over hyperparameters.

        Delegates to :func:`hyperparams.sensitivity_analysis`.

        Returns
        -------
        SensitivityResult
        """
        from .hyperparams import sensitivity_analysis
        return sensitivity_analysis(
            self, X, y=y, depth_range=depth_range,
            min_samples_leaf_range=min_samples_leaf_range,
            n_folds=n_folds, random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_surrogate(
        is_regression: bool,
        *,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        ccp_alpha: float = 0.0,
        monotonic_cst: Optional[np.ndarray] = None,
        random_state: int = 42,
    ):
        """Instantiate a configured sklearn surrogate tree."""
        kwargs: dict = dict(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )
        if monotonic_cst is not None:
            kwargs["monotonic_cst"] = monotonic_cst
        if is_regression:
            return DecisionTreeRegressor(**kwargs)
        return DecisionTreeClassifier(**kwargs)

    def _build_report(
        self,
        surrogate,
        X: np.ndarray,
        y_bb: np.ndarray,
        y_true: Optional[np.ndarray],
        ruleset: RuleSet,
        *,
        evaluation_type: str = "in_sample",
        avg_conditions_per_feature: Optional[float] = None,
        interaction_strength: Optional[float] = None,
    ) -> FidelityReport:
        """Build a FidelityReport for either classification or regression."""
        if self._task == "regression":
            return compute_regression_fidelity_report(
                surrogate=surrogate,
                X=X,
                y_bb=y_bb,
                y_true=y_true,
                num_rules=ruleset.num_rules,
                avg_rule_length=ruleset.avg_conditions,
                max_rule_length=ruleset.max_conditions,
                evaluation_type=evaluation_type,
                avg_conditions_per_feature=avg_conditions_per_feature,
                interaction_strength=interaction_strength,
            )
        return compute_fidelity_report(
            surrogate=surrogate,
            X=X,
            y_bb=y_bb,
            y_true=y_true,
            class_names=self.class_names or (),
            num_rules=ruleset.num_rules,
            avg_rule_length=ruleset.avg_conditions,
            max_rule_length=ruleset.max_conditions,
            evaluation_type=evaluation_type,
            avg_conditions_per_feature=avg_conditions_per_feature,
            interaction_strength=interaction_strength,
        )

    def _extract_rules_from_tree(
        self, tree, *, is_regression: bool = False,
        feature_names_override: Optional[tuple[str, ...]] = None,
    ) -> list[Rule]:
        """Depth-first walk over the sklearn tree internals."""
        tree_ = tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold
        children_left = tree_.children_left
        children_right = tree_.children_right
        value = tree_.value  # shape (n_nodes, 1, n_classes) or (n_nodes, 1, 1)

        feat_names = feature_names_override or self.feature_names
        class_names = self.class_names or ()
        rules: list[Rule] = []

        def _dfs(node_id: int, conditions: list[Condition]) -> None:
            # Leaf node
            if children_left[node_id] == children_right[node_id]:
                if is_regression:
                    pred_value = float(value[node_id, 0, 0])
                    total = int(tree_.n_node_samples[node_id])
                    rules.append(
                        Rule(
                            conditions=tuple(conditions),
                            prediction=f"{pred_value:.4f}",
                            samples=total,
                            confidence=1.0,
                            leaf_id=node_id,
                            prediction_value=pred_value,
                        )
                    )
                else:
                    class_counts = value[node_id, 0]
                    predicted_class = int(np.argmax(class_counts))

                    real_total = int(tree_.n_node_samples[node_id])

                    weight_total = class_counts.sum()
                    confidence = float(class_counts[predicted_class] / weight_total) if weight_total > 0 else 0.0

                    class_name = (
                        class_names[predicted_class]
                        if predicted_class < len(class_names)
                        else str(predicted_class)
                    )
                    rules.append(
                        Rule(
                            conditions=tuple(conditions),
                            prediction=class_name,
                            samples=real_total,
                            confidence=confidence,
                            leaf_id=node_id,
                        )
                    )
                return

            feat_name = (
                feat_names[feature[node_id]]
                if feature[node_id] < len(feat_names)
                else f"feature_{feature[node_id]}"
            )
            thresh = float(threshold[node_id])

            # Left child: feature <= threshold
            _dfs(
                children_left[node_id],
                conditions + [Condition(feat_name, "<=", thresh)],
            )
            # Right child: feature > threshold
            _dfs(
                children_right[node_id],
                conditions + [Condition(feat_name, ">", thresh)],
            )

        _dfs(0, [])
        return rules
