import json
import os
from pathlib import Path

import joblib

MODELS = Path("models")


def test_metadata_exists():
    assert (MODELS / "metadata.json").exists(), "metadata.json not found"


def test_model_loads():
    meta = json.load(open(MODELS / "metadata.json", encoding="utf-8"))
    model_path = meta["model_file"]
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    model = joblib.load(model_path)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_predicts_on_simple_texts():
    meta = json.load(open(MODELS / "metadata.json", encoding="utf-8"))
    model = joblib.load(meta["model_file"])
    X = [
        "You are great person with charming personality and good attitude.",
        "You are the worst person i ever met, you idiot!",
    ]
    preds = model.predict(X)
    assert len(preds) == 2
