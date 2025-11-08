import json
import os
from pathlib import Path
import joblib
from app.utils import clean_text, logger

MODELS = Path("models")
_METADATA_PATH = MODELS / "metadata.json"

THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))
LOW_CONF_FLOOR = float(os.getenv("LOW_CONF_FLOOR", "0.65"))
SHORT_LEN = int(os.getenv("SHORT_LEN", "8"))

_model = None
_model_meta = {}
MODEL_VERSION = "v1"
_APPLY_CLEAN = None
_MODEL_THRESHOLD = THRESHOLD


def load_model():
    global _model, MODEL_VERSION, _model_meta, _APPLY_CLEAN, _MODEL_THRESHOLD
    meta = json.load(open(_METADATA_PATH, encoding="utf-8"))
    _model_meta = meta
    MODEL_VERSION = meta.get("created", meta.get("created_at", "v1"))

    if "threshold" in meta:
        _MODEL_THRESHOLD = float(meta["threshold"])
    else:
        _MODEL_THRESHOLD = THRESHOLD

    notes = str(meta.get("notes", "")).lower()
    preprocess = meta.get("preprocess", {})
    if isinstance(preprocess, dict):
        _APPLY_CLEAN = bool(preprocess.get("clean_text", "clean_text" in notes))
    else:
        _APPLY_CLEAN = "clean_text" in notes

    model_file = Path(meta["model_file"])
    if not model_file.exists():
        model_file = MODELS / meta["model_file"]

    logger.info("Loading model_file=%s meta=%s", model_file, meta)
    _model = joblib.load(model_file)
    logger.info("Model loaded type=%s", type(_model))
    logger.info("Model threshold=%s apply_clean=%s", _MODEL_THRESHOLD, _APPLY_CLEAN)

    try:
        sample = "you are idiot"
        input_for_model = clean_text(sample) if _APPLY_CLEAN else sample
        if hasattr(_model, "predict_proba"):
            p = float(_model.predict_proba([input_for_model])[0][1])
        else:
            p = float(_model.predict([input_for_model])[0])
        logger.info(
            "Smoke test sample='%s' -> prob=%.4f (applied_clean=%s)",
            sample,
            p,
            _APPLY_CLEAN,
        )
    except Exception as e:
        logger.exception("Smoke test failed: %s", e)


def predict_one(text: str):
    if _model is None:
        load_model()

    raw = text
    cleaned = clean_text(text)

    if _APPLY_CLEAN is not None:
        use_clean = _APPLY_CLEAN
    else:
        use_clean = not hasattr(_model, "named_steps")

    input_text = cleaned if use_clean else raw

    logger.debug(
        "Predicting (len=%d) use_clean=%s input=%s",
        len(text),
        use_clean,
        input_text[:200],
    )

    proba = float(_model.predict_proba([input_text])[0][1])
    label = "toxic" if proba >= _MODEL_THRESHOLD else "clean"
    low_confidence = (proba < max(_MODEL_THRESHOLD, LOW_CONF_FLOOR)) or (
        len(cleaned) < SHORT_LEN
    )

    logger.info(
        "predict len=%d label=%s prob=%.3f low_conf=%s",
        len(text),
        label,
        proba,
        low_confidence,
    )
    return {"label": label, "prob": proba, "low_confidence": low_confidence}
