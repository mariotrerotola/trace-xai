# Getting Started

## Requirements

- Python >= 3.9
- scikit-learn >= 1.3
- numpy >= 1.24
- matplotlib >= 3.7

## Installation

### From PyPI

```bash
pip install trace-xai
```

### From source (development)

```bash
git clone https://github.com/your-username/trace-xai.git
cd trace-xai
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# Graphviz DOT export support
pip install trace-xai[graphviz]

# Comparative benchmark (LIME + SHAP)
pip install trace-xai[benchmark]

# GAM surrogate (future)
pip install trace-xai[gam]
```

## Your First Explanation in 60 Seconds

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from trace_xai import Explainer

# 1. Train any model
iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# 2. Create an Explainer
explainer = Explainer(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
)

# 3. Extract rules
result = explainer.extract_rules(iris.data, y=iris.target, max_depth=4)

# 4. Read the rules
print(result.rules)

# 5. Read the fidelity report
print(result.report)

# 6. Visualize the surrogate tree
result.plot(save_path="my_tree.png")
```

**Output:**

```
Rule 1: IF petal_length <= 2.4500 THEN class = setosa  [confidence=100.00%, samples=50]
Rule 2: IF petal_length > 2.4500 AND petal_width <= 1.7500 AND petal_length <= 4.9500
        THEN class = versicolor  [confidence=97.73%, samples=44]
...

=== Fidelity Report ===
  Evaluation type: in_sample
  Fidelity (surrogate vs black-box): 0.9800
  Surrogate accuracy (vs true labels): 0.9733
  Black-box accuracy (vs true labels): 1.0000
  Number of rules: 7
  Avg rule length: 2.86
  ...
```

## Understanding the Output

### Rules

Each rule is an IF-THEN statement extracted from the leaves of the surrogate
decision tree:

```
IF petal_length <= 2.4500 THEN class = setosa  [confidence=100.00%, samples=50]
```

- **IF ... THEN** — the logical path from the tree root to a leaf.
- **confidence** — what fraction of training samples at that leaf agree on the
  predicted class.
- **samples** — how many training samples landed in that leaf.

### Fidelity Report

The report quantifies how well the simple surrogate mimics the complex
black-box:

| Metric | Meaning |
|--------|---------|
| **Fidelity** | Agreement rate between surrogate and black-box predictions. A fidelity of 0.98 means the surrogate agrees with the black-box on 98% of samples. |
| **Accuracy** | Surrogate accuracy against the *true* labels (only if `y` is provided). |
| **Black-box accuracy** | Original model accuracy against true labels. |
| **Number of rules** | Total leaves in the surrogate tree = total rules. |
| **Avg/Max rule length** | Conditions per rule — a proxy for rule complexity. |
| **Per-class fidelity** | Fidelity broken down by predicted class. |

## Next Steps

- Read the [User Guide](user_guide.md) for hold-out evaluation, cross-validation,
  stability analysis, and regression.
- Read the [API Reference](api_reference.md) for complete method signatures.
- Read the [Methodology](methodology.md) for the scientific background.
