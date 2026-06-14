"""Train an XGBoost classifier with class weighting and early stopping."""
import numpy as np
import joblib
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report

from ml.config import SEED, XGB_MODEL_FILE
from ml.preprocessing.pipeline import prepare, load_dataset


def train():
    X_tr, X_val, X_te, y_tr, y_val, y_te, _ = prepare(load_dataset())

    le = LabelEncoder()
    y_tr_e = le.fit_transform(y_tr)
    y_val_e = le.transform(y_val)
    y_te_e = le.transform(y_te)

    sample_weight = compute_sample_weight("balanced", y_tr_e)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        callbacks=[EarlyStopping(rounds=20, save_best=True)],
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_tr, y_tr_e,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val_e)],
        verbose=False,
    )

    preds = le.inverse_transform(model.predict(X_te))
    print(classification_report(y_te, preds))

    joblib.dump({"model": model, "label_encoder": le}, XGB_MODEL_FILE)
    print(f"Saved model to {XGB_MODEL_FILE}")


if __name__ == "__main__":
    train()
