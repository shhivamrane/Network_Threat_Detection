"""Evaluate trained models on the official NSL-KDD KDDTest+ hold-out.

KDDTest+ contains attack types absent from KDDTrain+, so scores are
realistically lower than the internal split — this is the hard benchmark.
"""
import json
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tensorflow as tf

from ml.config import (
    CLASSES, PREPROCESSOR_FILE, RF_MODEL_FILE, XGB_MODEL_FILE, DL_MODEL_FILE,
)
from ml.data import load_nslkdd


def run():
    df = load_nslkdd("test")
    df = df[df["attack_class"].isin(CLASSES)]
    X = df.drop(columns=["attack_class"])
    y = df["attack_class"]

    pre = joblib.load(PREPROCESSOR_FILE)
    Xt = pre.transform(X)

    def report(name, pred):
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro", labels=CLASSES)
        print(f"\n=== {name} (KDDTest+) ===")
        print(f"Accuracy: {acc:.4f}   Macro F1: {f1:.4f}")
        print(classification_report(y, pred, labels=CLASSES))

    rf = joblib.load(RF_MODEL_FILE)
    report("Random Forest", rf.predict(Xt))

    bundle = joblib.load(XGB_MODEL_FILE)
    xgb, le = bundle["model"], bundle["label_encoder"]
    report("XGBoost", le.inverse_transform(xgb.predict(Xt)))

    dl = tf.keras.models.load_model(DL_MODEL_FILE)
    dl_classes = np.array(json.loads((DL_MODEL_FILE.parent / "dl_classes.json").read_text()))
    report("Deep Learning", dl_classes[np.argmax(dl.predict(Xt, verbose=0), axis=1)])


if __name__ == "__main__":
    run()