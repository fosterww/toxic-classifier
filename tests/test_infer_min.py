import json
import os

import joblib


def test_model_exists_and_loads():
    assert os.path.exists("models/metadata.json")
    meta = json.load(open("models/metadata.json", encoding="utf-8"))
    assert os.path.exists(meta["model_file"])
    model = joblib.load(meta["model_file"])
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
