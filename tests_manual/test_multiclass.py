import sys
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add src to python path to import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from trace_xai.explainer import Explainer

def test_multiclass_complete():
    print("==================================================")
    print("STARTING COMPLETE MULTI-CLASS CLASSIFICATION TEST")
    print("==================================================")

    # 1. Load Data
    print("\n1. Loading Iris Data (3 Classes)...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = list(iris.target_names)
    print(f"   Classes: {class_names}")
    print(f"   Features: {feature_names}")
    print(f"   Shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train Black-Box Model
    print("\n2. Training Black-Box Model (RandomForestClassifier)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   Black-Box Model Accuracy on Test Set: {rf_acc:.4f}")

    # 3. Initialize Explainer
    print("\n3. Initializing Explainer...")
    explainer = Explainer(
        model=rf,
        feature_names=feature_names,
        class_names=class_names,
        task="classification"
    )

    # 4. Extract Rules (Complete Test)
    print("\n4. Extracting Rules...")
    # Using hold-out validation (X_val) explicitly
    result = explainer.extract_rules(
        X_train,
        y=y_train,
        X_val=X_test,
        y_val=y_test
    )

    print("\n   --- Extracted Rules ---")
    print(result.rules)

    print("\n   --- Fidelity Report ---")
    print(result.report)
    
    if result.train_report:
        print("\n   --- Training Fidelity Report ---")
        print(result.train_report)

    # 5. Cross-Validation Fidelity
    print("\n5. Running Cross-Validation Fidelity Check (5 folds)...")
    cv_report = explainer.cross_validate_fidelity(
        X, 
        y=y, 
        n_folds=5, 
        random_state=42
    )
    print(f"   Mean Fidelity: {cv_report.mean_fidelity:.4f} (+/- {cv_report.std_fidelity:.4f})")
    print(f"   Mean Accuracy: {cv_report.mean_accuracy:.4f} (+/- {cv_report.std_accuracy:.4f})")

    # 6. Stability Analysis
    print("\n6. Running Stability Analysis (10 bootstraps)...")
    stability = explainer.compute_stability(
        X, 
        n_bootstraps=10, 
        random_state=42
    )
    print(f"   Mean Jaccard Similarity: {stability.mean_jaccard:.4f} (+/- {stability.std_jaccard:.4f})")

    # 7. Confidence Intervals
    print("\n7. Computing Confidence Intervals...")
    ci = explainer.compute_confidence_intervals(
        result, 
        X_test, 
        y=y_test, 
        n_bootstraps=100
    )
    print(f"   Fidelity 95% CI: {ci['fidelity'].lower:.4f} - {ci['fidelity'].upper:.4f}")
    if 'accuracy' in ci:
        print(f"   Accuracy 95% CI: {ci['accuracy'].lower:.4f} - {ci['accuracy'].upper:.4f}")

    # 8. HTML Export
    html_path = os.path.join(current_dir, "multiclass_report.html")
    print(f"\n8. Exporting HTML Report to {html_path}...")
    try:
        result.to_html(html_path)
        print("   Report saved successfully.")
    except Exception as e:
        print(f"   Failed to save report: {e}")

    print("\n==================================================")
    print("TEST COMPLETED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    test_multiclass_complete()
