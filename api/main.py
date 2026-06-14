"""FastAPI inference service for the threat-detection system."""
import json
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ml.config import PREPROCESSOR_FILE, XGB_MODEL_FILE, METADATA_FILE
from api.schemas import BatchRequest, FlowFeatures, Prediction, ModelInfo
from api import db

# Loaded once at startup.
PRE = None
MODEL = None
LE = None
META = None
CONNECTIONS: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PRE, MODEL, LE, META
    db.init_db()
    PRE = joblib.load(PREPROCESSOR_FILE)
    bundle = joblib.load(XGB_MODEL_FILE)  # XGBoost as the production model
    MODEL, LE = bundle["model"], bundle["label_encoder"]
    META = json.loads(METADATA_FILE.read_text())
    yield


app = FastAPI(title="AI Threat Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _predict_df(df: pd.DataFrame) -> list[Prediction]:
    X = PRE.transform(df)
    proba = MODEL.predict_proba(X)
    idx = np.argmax(proba, axis=1)
    labels = LE.inverse_transform(idx)
    out = []
    for label, p in zip(labels, proba):
        out.append(Prediction(
            attack_class=str(label),
            confidence=float(np.max(p)),
            is_attack=(str(label) != "Normal"),
        ))
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    return ModelInfo(model="XGBoost", classes=META["classes"],
                     n_features=META["n_features"])


@app.post("/predict", response_model=Prediction)
async def predict(req: FlowFeatures):
    df = pd.DataFrame([req.features])
    pred = _predict_df(df)[0]
    db.save_detection(pred.attack_class, pred.confidence, pred.is_attack, req.features)
    await _broadcast(pred)
    return pred


@app.post("/predict/batch", response_model=list[Prediction])
async def predict_batch(req: BatchRequest):
    df = pd.DataFrame(req.flows)
    preds = _predict_df(df)
    for pred, feats in zip(preds, req.flows):
        db.save_detection(pred.attack_class, pred.confidence, pred.is_attack, feats)
        await _broadcast(pred)
    return preds


@app.get("/detections")
def detections(limit: int = 100):
    return db.recent(limit)


async def _broadcast(pred: Prediction):
    dead = []
    for ws in CONNECTIONS:
        try:
            await ws.send_json(pred.model_dump())
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        CONNECTIONS.remove(ws)


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    CONNECTIONS.append(ws)
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"keepalive": True})
    except WebSocketDisconnect:
        CONNECTIONS.remove(ws)