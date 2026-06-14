"""Pydantic schemas for the inference API."""
from pydantic import BaseModel
from typing import Any


class FlowFeatures(BaseModel):
    """A single flow as a feature dict (NSL-KDD schema)."""
    features: dict[str, Any]


class BatchRequest(BaseModel):
    flows: list[dict[str, Any]]


class Prediction(BaseModel):
    attack_class: str
    confidence: float
    is_attack: bool


class ModelInfo(BaseModel):
    model: str
    classes: list[str]
    n_features: int