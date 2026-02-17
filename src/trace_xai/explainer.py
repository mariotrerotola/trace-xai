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
    ) -> None:
        self.rules = rules
        self.report = report
        self.surrogate = surrogate
        self._feature_names = feature_names
        self._class_names = class_names or ()
        self.train_report = train_report
        self._task = task

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
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        validation_split: Optional[float] = None,
        surrogate_type: str = "decision_tree",
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
        X_val : array-like, optional
            Separate validation set for hold-out fidelity evaluation.
        y_val : array-like, optional
            True labels for the validation set.
        validation_split : float, optional
            If given (0 < value < 1), split *X* internally for hold-out
            evaluation.  Mutually exclusive with *X_val*.

        Returns
        -------
        ExplanationResult
        """
        if surrogate_type != "decision_tree":
            raise ValueError(
                f"Only 'decision_tree' surrogate is currently supported, "
                f"got {surrogate_type!r}. See surrogates/ package for placeholders."
            )

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

        # 2. Train surrogate
        is_regression = self._task == "regression"
        if is_regression:
            surrogate = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
        else:
            surrogate = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
        surrogate.fit(X_train, y_bb)

        # 3. Extract rules via DFS
        rules = self._extract_rules_from_tree(surrogate, is_regression=is_regression)
        ruleset = RuleSet(
            rules=tuple(rules),
            feature_names=self.feature_names,
            class_names=self.class_names or (),
        )

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
            if is_regression:
                surr = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                )
            else:
                surr = DecisionTreeClassifier(
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
    ) -> StabilityReport:
        """Compute rule stability via bootstrap resampling + Jaccard similarity."""
        X = np.asarray(X)
        rng = np.random.RandomState(random_state)

        signatures: list[frozenset] = []
        is_regression = self._task == "regression"

        for _ in range(n_bootstraps):
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            y_bb_boot = np.asarray(self.model.predict(X_boot))

            if is_regression:
                surr = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                )
            else:
                surr = DecisionTreeClassifier(
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
    ) -> list[Rule]:
        """Depth-first walk over the sklearn tree internals."""
        tree_ = tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold
        children_left = tree_.children_left
        children_right = tree_.children_right
        value = tree_.value  # shape (n_nodes, 1, n_classes) or (n_nodes, 1, 1)

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
                    total = int(class_counts.sum())
                    confidence = float(class_counts[predicted_class] / total) if total else 0.0
                    class_name = (
                        class_names[predicted_class]
                        if predicted_class < len(class_names)
                        else str(predicted_class)
                    )
                    rules.append(
                        Rule(
                            conditions=tuple(conditions),
                            prediction=class_name,
                            samples=total,
                            confidence=confidence,
                            leaf_id=node_id,
                        )
                    )
                return

            feat_name = (
                self.feature_names[feature[node_id]]
                if feature[node_id] < len(self.feature_names)
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
