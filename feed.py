"""Send synthetic flows to the running API so the dashboard populates.
Run from project root with the API already up:  python feed.py
"""
import json
import time
import urllib.request

from ml.data import make_synthetic

URL = "http://127.0.0.1:8000/predict"


def to_native(row):
    # JSON can't serialize numpy scalars; unwrap them.
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def send(features):
    body = json.dumps({"features": features}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    flows = make_synthetic(80).drop(columns=["attack_class"])
    for _, row in flows.iterrows():
        pred = send(to_native(row.to_dict()))
        flag = "ATTACK" if pred["is_attack"] else "ok"
        print(f"{pred['attack_class']:<7} {pred['confidence']:.2f} {flag}")
        time.sleep(0.4)