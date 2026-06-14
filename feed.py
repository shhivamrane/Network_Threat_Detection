"""Send real NSL-KDD test flows to the running API so the dashboard populates.

Uses official KDDTest+ records (the schema the model was trained on), so the
live demo reflects the model's real behavior. Falls back to synthetic if
NSL-KDD isn't present. Run from project root with the API up:  python feed.py
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/predict"


def to_native(row):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def send(features):
    body = json.dumps({"features": features}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def load_flows(n=120):
    try:
        from ml.data import load_nslkdd
        df = load_nslkdd("test").sample(frac=1).reset_index(drop=True)
        print(f"Using real NSL-KDD KDDTest+ ({len(df)} rows available).")
        return df.head(n)
    except FileNotFoundError:
        from ml.data import make_synthetic
        print("NSL-KDD not found - using synthetic data.")
        return make_synthetic(n)


if __name__ == "__main__":
    flows = load_flows(120)
    correct = total = 0
    for _, row in flows.iterrows():
        actual = row.get("attack_class", "?")
        features = to_native(row.drop(labels=["attack_class"]).to_dict())
        pred = send(features)
        total += 1
        hit = pred["attack_class"] == actual
        correct += hit
        mark = "OK  " if hit else "MISS"
        print(f"{mark} pred={pred['attack_class']:<7} actual={actual:<7} conf={pred['confidence']:.2f}")
        time.sleep(0.3)
    if total:
        print(f"\nLive accuracy: {correct}/{total} = {correct/total:.1%}")
