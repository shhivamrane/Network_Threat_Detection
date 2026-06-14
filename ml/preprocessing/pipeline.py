"""Reusable preprocessing pipeline shared by all models and the API."""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from ml.config import SEED, PREPROCESSOR_FILE, METADATA_FILE, CLASSES


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, replace Inf, and remove constant columns."""
    df = df.drop_duplicates().copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    nunique = df.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    constant = [c for c in constant if c != "attack_class"]
    return df.drop(columns=constant)


def build_preprocessor(df: pd.DataFrame):
    """Construct a ColumnTransformer for the given feature frame."""
    features = df.drop(columns=["attack_class"])
    cat_cols = features.select_dtypes(include="object").columns.tolist()
    num_cols = features.select_dtypes(include="number").columns.tolist()

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return pre, num_cols, cat_cols


def prepare(df: pd.DataFrame, use_smote: bool = True):
    """Clean, split, fit the preprocessor, and return ready-to-train arrays."""
    df = _clean(df)
    df = df[df["attack_class"].isin(CLASSES)]

    X = df.drop(columns=["attack_class"])
    y = df["attack_class"]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )

    pre, num_cols, cat_cols = build_preprocessor(df)
    X_tr_t = pre.fit_transform(X_tr)
    X_val_t = pre.transform(X_val)
    X_te_t = pre.transform(X_te)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            min_count = y_tr.value_counts().min()
            k = max(1, min(5, min_count - 1))
            X_tr_t, y_tr = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(
                X_tr_t, y_tr
            )
        except Exception as e:  # noqa: BLE001
            print(f"SMOTE skipped: {e}")

    joblib.dump(pre, PREPROCESSOR_FILE)
    feature_names = list(pre.get_feature_names_out())
    METADATA_FILE.write_text(json.dumps({
        "classes": CLASSES,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }, indent=2))

    return X_tr_t, X_val_t, X_te_t, y_tr, y_val, y_te, pre


def load_dataset(use_synthetic: bool = True) -> pd.DataFrame:
    """Load NSL-KDD if present, else fall back to synthetic data."""
    try:
        from ml.data import load_nslkdd
        return load_nslkdd("train")
    except FileNotFoundError:
        from ml.data import make_synthetic
        print("Real dataset not found - using synthetic data.")
        return make_synthetic()
