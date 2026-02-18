
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def get_model(model_name, task, random_state=42):
    """
    Get a model instance based on name and task.

    Args:
        model_name (str): 'rf', 'xgb', 'mlp'
        task (str): 'classification'
        random_state (int): Seed for reproducibility

    Returns:
        model: Sklearn-compatible model
    """
    if task != 'classification':
        raise ValueError(f"Only classification is supported, got: {task}")

    if model_name == 'rf':
        return RandomForestClassifier(n_estimators=500, random_state=random_state)
    elif model_name == 'xgb':
        return XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(128, 128), random_state=random_state, max_iter=500)
    else:
        raise ValueError(f"Unknown model: {model_name}")
