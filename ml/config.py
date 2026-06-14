from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

MODEL_DIR.mkdir(exist_ok=True)

# 5-class attack taxonomy used across the whole project
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Artifact filenames
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"
RF_MODEL_FILE = MODEL_DIR / "random_forest.joblib"
XGB_MODEL_FILE = MODEL_DIR / "xgboost.joblib"
DL_MODEL_FILE = MODEL_DIR / "deep_model.keras"
METADATA_FILE = MODEL_DIR / "metadata.json"
