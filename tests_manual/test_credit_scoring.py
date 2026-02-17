
import sys
import os
import numpy as np
import pandas as pd
import time
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Add src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from trace_xai.explainer import Explainer

def generate_credit_data(n_samples=5000):
    """
    Generates a synthetic 'Credit Scoring' dataset with realistic feature names 
    and non-linear relationships to simulate a complex structured data problem.
    """
    np.random.seed(42)
    
    # 1. Generate core features
    data = {
        'Income': np.random.gamma(shape=2, scale=20000, size=n_samples) + 20000,  # Skewed income
        'Credit_Score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'Age': np.random.uniform(18, 70, n_samples),
        'Debt_to_Income': np.random.uniform(0, 1, n_samples),
        'Loan_Amount': np.random.exponential(15000, n_samples) + 1000,
        'Num_Credit_Cards': np.random.poisson(3, n_samples),
        'Years_Employed': np.random.exponential(5, n_samples),
        'Missed_Payments': np.random.poisson(0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 2. Define complex Logic for "Risk Level" (0=Low, 1=Medium, 2=High)
    # This ground truth is unknown to the model, which must learn it.
    
    risk_score = np.zeros(n_samples)
    
    # Base logic
    risk_score += (df['Income'] < 40000) * 20
    risk_score += (df['Credit_Score'] < 600) * 30
    risk_score += (df['Credit_Score'] > 750) * -20
    risk_score += (df['Debt_to_Income'] > 0.4) * 25
    risk_score += (df['Missed_Payments'] > 0) * 40
    
    # Non-linear interactions (The "Complex" part)
    # Young people with high loans are risky regardless of income
    risk_score += ((df['Age'] < 25) & (df['Loan_Amount'] > 20000)) * 25
    
    # High income but low credit score = erratic behavior
    risk_score += ((df['Income'] > 100000) & (df['Credit_Score'] < 650)) * 15
    
    # Stable employment reduces risk
    risk_score -= (df['Years_Employed'] > 5) * 10

    # Add noise to make it realistic (perfect rules don't exist in reality)
    risk_score += np.random.normal(0, 10, n_samples)
    
    # Convert score to classes
    # Low Risk (Top 33%), Medium (Mid 33%), High (Bottom 33%)
    t1, t2 = np.percentile(risk_score, [33, 66])
    
    def get_class(score):
        if score < t1: return 0 # Low Risk
        if score < t2: return 1 # Medium Risk
        return 2 # High Risk
        
    y = np.array([get_class(s) for s in risk_score])
    
    class_names = ["Low_Risk", "Medium_Risk", "High_Risk"]
    
    return df, y, class_names

def test_structured_credit_scoring():
    print("==========================================================================")
    print("   COMPLEX STRUCTURED DATA: Credit Risk Scoring (Synthetic Financial Data)")
    print("==========================================================================")

    # 1. Load Data
    print("\n[1] Generating Financial Dataset...")
    df, y, class_names = generate_credit_data(n_samples=5000)
    
    feature_names = df.columns.tolist()
    X = df.values
    
    print(f"    Samples: {len(df)}")
    print(f"    Features: {len(feature_names)} {feature_names}")
    print(f"    Classes: {class_names}")
    print("    Shape: ", X.shape)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train Complex Model
    # Histogram-based Gradient Boosting is similar to LightGBM/XGBoost - State of the Art for Tabular
    print("\n[2] Training Black-Box Model (HistGradientBoostingClassifier)...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"    Training took {time.time() - start_time:.2f}s")
    
    # Evaluate BB
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"    Black-Box Accuracy: {acc:.4f}")
    print("    Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 3. Explain with TRACE
    print("\n[3] Initializing Explainer...")
    explainer = Explainer(
        model=model,
        feature_names=feature_names,
        class_names=class_names,
        task="classification"
    )

    # 4. Extract Rules (Standard vs Stable)
    print("\n[4a] Standard Extraction (Baseline)...")
    result_std = explainer.extract_rules(
        X_train, 
        y=y_train, 
        X_val=X_test, 
        y_val=y_test,
        max_depth=4,
        min_samples_leaf=20
    )
    print(f"    Standard Fidelity: {result_std.report.fidelity:.2%}")

    print("\n[4b] Regulatory Stable Extraction (New Feature)...")
    # Define tolerances based on feature scales
    tolerances = {
        'Income': 5000.0,           # $5k buckets
        'Credit_Score': 25.0,       # 25 points
        'Age': 5.0,                 # 5 years
        'Debt_to_Income': 0.1,      # 10%
        'Loan_Amount': 2000.0,      # $2k
        'Num_Credit_Cards': 1.0,    # 1 card
        'Years_Employed': 2.0,      # 2 years
        'Missed_Payments': 0.5,     # Exact match (since integer)
    }

    # Using the new extract_stable_rules method
    result_stable = explainer.extract_stable_rules(
        X_train,
        y=y_train,
        n_estimators=30,          # Train 30 surrogates for better stats
        frequency_threshold=0.6,  # Rules must appear in 60% of trees
        tolerance=tolerances,     # Feature-specific matching
        max_depth=4,
        min_samples_leaf=20,
        X_val=X_test,
        y_val=y_test
    )

    print("\n    --- Stable Fidelity Report ---")
    print(f"    Fidelity: {result_stable.report.fidelity:.2%}")
    print(f"    Ensemble Report: {result_stable.ensemble_report}")
    
    print("\n    --- Top Stable Rules (Regulatory Compliant) ---")
    # Sort by frequency to show the most robust rules first
    sorted_stable = sorted(result_stable.stable_rules, key=lambda r: r.frequency, reverse=True)
    
    for i, sr in enumerate(sorted_stable[:5]):
        print(f"    Rule {i+1} [Freq: {sr.frequency:.0%}, Conf: {sr.rule.confidence:.0%}]: {sr.rule}")

    # 4c. Apply Pruning
    from trace_xai.pruning import PruningConfig
    print("\n[4c] Applying Regulatory Pruning...")
    pruning_config = PruningConfig(
        min_confidence=0.8,       # Only high confidence rules
        min_samples=1,            # Relaxed for investigation
        remove_redundant=True     # Simplify logic
    )
    result_pruned = explainer.prune_rules(result_stable, pruning_config)
    print(f"    Pruned Rules Count: {result_pruned.pruned_rules.num_rules} (from {result_stable.rules.num_rules})")
    print(f"    Pruning Report: {result_pruned.pruning_report}")

    # 5. Stability Re-Check
    # Note: computed stability of the *process* of finding stable rules is meta-stability.
    # Here we just show the result of the ensemble process which IS the stability solution.
    print(f"\n    Stable Rules Found: {len(result_stable.stable_rules)}")

    # 6. HTML Export
    html_path = os.path.join(current_dir, "credit_scoring_stable_report.html")
    print(f"\n[6] Exporting Report to {html_path}")
    result_stable.to_html(html_path)
    print("    Report generated.")

if __name__ == "__main__":
    test_structured_credit_scoring()
