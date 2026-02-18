"""Dataset loading with in-process cache for parallel experiments."""

import functools

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder


@functools.lru_cache(maxsize=None)
def load_dataset(name):
    """Load dataset by name (cached after first call).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series | np.ndarray
    task : str
    feature_names : list[str]
    categorical_features : list[str]
    """
    if name == "adult":
        X, y = fetch_openml(data_id=1590, return_X_y=True, as_frame=True)
        df = pd.concat([X, y], axis=1).dropna()
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1]
        cat_features = X.select_dtypes(include=["category", "object"]).columns.tolist()
        for col in cat_features:
            X[col] = LabelEncoder().fit_transform(X[col])
        y = LabelEncoder().fit_transform(y)
        return X, y, "classification", list(X.columns), cat_features

    if name == "german_credit":
        X, y = fetch_openml(data_id=31, return_X_y=True, as_frame=True)
        cat_features = X.select_dtypes(include=["category", "object"]).columns.tolist()
        for col in cat_features:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        y = LabelEncoder().fit_transform(y)
        return X, y, "classification", list(X.columns), cat_features

    if name == "compas":
        url = (
            "https://raw.githubusercontent.com/propublica/compas-analysis/"
            "master/compas-scores-two-years.csv"
        )
        try:
            df = pd.read_csv(url)
            df = df[
                (df["days_b_screening_arrest"] <= 30)
                & (df["days_b_screening_arrest"] >= -30)
                & (df["is_recid"] != -1)
                & (df["c_charge_degree"] != "O")
                & (df["score_text"] != "N/A")
            ]
            features = [
                "age", "c_charge_degree", "race", "sex",
                "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count",
            ]
            X = df[features].copy()
            y = df["two_year_recid"].values
            for col in ["c_charge_degree", "race", "sex"]:
                X[col] = LabelEncoder().fit_transform(X[col])
            return X, pd.Series(y), "classification", list(X.columns), ["c_charge_degree", "race", "sex"]
        except Exception as e:
            print(f"Failed to load COMPAS: {e}")
            return None, None, None, None, None

    raise ValueError(f"Unknown dataset: {name}")
