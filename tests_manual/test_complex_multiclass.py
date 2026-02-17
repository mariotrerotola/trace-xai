
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

# Add src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from trace_xai.explainer import Explainer

def test_complex_multiclass():
    print("=====================================================================")
    print("   COMPLEX SCENARIO: 10 Classes, High Dimensionality, Non-Linear     ")
    print("=====================================================================")

    # 1. Load Digits Dataset (Complex Real-World Data)
    print("\n[1] Loading Digits Dataset (Images)...")
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    class_names = [f"Digit_{i}" for i in range(10)]
    
    print(f"    Samples: {X.shape[0]}")
    print(f"    Features: {X.shape[1]} (8x8 pixel intensity)")
    print(f"    Classes: {len(class_names)}")

    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Train a "True" Black-Box Model (Neural Network)
    print("\n[2] Training Black-Box Model (MLPClassifier / Neural Net)...")
    # A neural network with non-linear activations is a classic black box
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    start_time = time.time()
    mlp.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    bb_acc = accuracy_score(y_test, mlp.predict(X_test_scaled))
    print(f"    Training completed in {train_time:.2f}s")
    print(f"    Black-Box Test Accuracy: {bb_acc:.4f} (This is the target to explain)")

    # 3. Initialize Explainer
    print("\n[3] Initializing TRACE Explainer...")
    explainer = Explainer(
        model=mlp,
        feature_names=feature_names,
        class_names=class_names,
        task="classification"
    )

    # 4. Extract Rules with Deeper Surrogate
    # Since the problem is complex, a depth=3 tree won't suffice. 
    # We increase depth to 7 or 8 to capture more nuance, trading off some interpretability for fidelity.
    target_depth = 7
    print(f"\n[4] Extracting Rules (Surrogate Depth={target_depth})...")
    
    result = explainer.extract_rules(
        X_train_scaled,
        y=y_train,
        X_val=X_test_scaled,
        y_val=y_test,
        max_depth=target_depth,
        min_samples_leaf=10  # Reduced overfitting of surrogate
    )

    print("\n    --- Fidelity Report ---")
    print(f"    Fidelity (Surrogate mimics BB): {result.report.fidelity:.4f}")
    print(f"    Surrogate Accuracy (Ground Truth): {result.report.accuracy:.4f}")
    print(f"    Number of Rules: {result.report.num_rules}")
    print(f"    Avg Rule Length: {result.report.avg_rule_length:.2f}")

    # Show top 3 rules just to see format
    print("\n    --- Example Rules (First 3) ---")
    for i, rule in enumerate(result.rules.rules[:3]):
        print(f"    Rule {i+1}: {rule}")

    # 5. Stability Analysis
    print("\n[5] Analyzing Stability (Robustness of Explanations)...")
    # Complex models often result in unstable trees. This metric is crucial here.
    stability = explainer.compute_stability(
        X_test_scaled,
        n_bootstraps=15, 
        max_depth=target_depth,
        random_state=42
    )
    print(f"    Mean Jaccard Index: {stability.mean_jaccard:.4f} (Higher is more stable)")

    # 6. HTML Export with Full Details
    html_filename = "complex_multiclass_report.html"
    html_path = os.path.join(current_dir, html_filename)
    print(f"\n[6] Exporting interactive report to {html_path}...")
    result.to_html(html_path)
    print("    Done.")

if __name__ == "__main__":
    test_complex_multiclass()
