"""Train a Keras MLP for multi-class intrusion detection."""
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from ml.config import SEED, DL_MODEL_FILE, MODEL_DIR
from ml.preprocessing.pipeline import prepare, load_dataset

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

tf.random.set_seed(SEED)


def build_model(n_features: int, n_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    X_tr, X_val, X_te, y_tr, y_val, y_te, _ = prepare(load_dataset())

    le = LabelEncoder()
    y_tr_e = le.fit_transform(y_tr)
    y_val_e = le.transform(y_val)
    y_te_e = le.transform(y_te)

    weights = compute_class_weight("balanced", classes=np.unique(y_tr_e), y=y_tr_e)
    class_weight = dict(enumerate(weights))

    model = build_model(X_tr.shape[1], len(le.classes_))
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        X_tr, y_tr_e, validation_data=(X_val, y_val_e),
        epochs=100, batch_size=256, class_weight=class_weight,
        callbacks=[es], verbose=0,
    )

    preds = le.inverse_transform(np.argmax(model.predict(X_te), axis=1))
    print(classification_report(y_te, preds))

    model.save(DL_MODEL_FILE)
    (MODEL_DIR / "dl_classes.json").write_text(json.dumps(le.classes_.tolist()))
    return model


if __name__ == "__main__":
    train()
