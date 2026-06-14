"""Train a tuned Random Forest classifier."""
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

from ml.config import SEED, RF_MODEL_FILE, MODEL_DIR
from ml.preprocessing.pipeline import prepare, load_dataset


def train():
    X_tr, X_val, X_te, y_tr, y_val, y_te, _ = prepare(load_dataset())

    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        param_dist, n_iter=10, cv=3, scoring="f1_macro",
        random_state=SEED, n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    model = search.best_estimator_

    print("Best params:", search.best_params_)
    print(classification_report(y_te, model.predict(X_te)))

    joblib.dump(model, RF_MODEL_FILE)
    (MODEL_DIR / "rf_best_params.json").write_text(json.dumps(search.best_params_))
    return model


if __name__ == "__main__":
    train()
