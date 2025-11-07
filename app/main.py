import csv
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthOut, PredictIn, PredictOut, FeedbackIn, FeedbackOut
from app.predict import predict_one, load_model, MODEL_VERSION
from app.utils import logger

ALLOWED_ORIGINS = os.getenv("CORS_ORIGIN", "*").split(",")

app = FastAPI(
    title="Toxicity API",
    version="1.0.0",
    description="Binary toxicity classifier (TF-IDF + Logistic Regression).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEED = Path("data/feedback.csv")
FEED.parent.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def _startup():
    try:
        load_model()
        logger.info("Startup: model ready")
    except Exception as e:
        logger.exception("Startup model load failed: %s", e)


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictOut, tags=["inference"])
def predict(payload: PredictIn):
    try:
        result = predict_one(payload.text)
        logger.info(
            "predict len=%d label=%s prob=%.3f low_conf=%s",
            len(payload.text),
            result["label"],
            result["prob"],
            result["low_confidence"],
        )
        return result
    except Exception as e:
        logger.exception("Predcit failed: %s", e)
        raise HTTPException(status_code=500, detail="internal error")


@app.post("/feedback", response_model=FeedbackOut, tags=["feedback"])
def feedback(item: FeedbackIn):
    try:
        with open(FEED, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([datetime.utcnow().isoformat(), item.text, item.true_label])
        return {"status": "stored"}
    except Exception as e:
        logger.exception("Feedback failed: %s", e)
        raise HTTPException(status_code=500, detail="internal error")
