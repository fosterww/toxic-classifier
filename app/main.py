import csv
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthOut, PredictIn, PredictOut, FeedbackIn, FeedbackOut
from app.predict import predict_one, load_model, MODEL_VERSION
from app.utils import logger
from app.db import SessionLocal
from app.db_models import Feedback, Prediction

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
    db = SessionLocal()
    try:
        result = predict_one(payload.text)
        prediction = Prediction(
            text=payload.text,
            pred_label=result["label"],
            prob=result["prob"],
        )
        db.add(prediction)
        db.commit()
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
    db = SessionLocal()
    try:
        f = Feedback(text=item.text, pred_label="?", true_label=item.true_label)
        db.add(f)
        db.commit()
        db.refresh(f)
        logger.info("feedback saved id=%d", f.id)
        return {"status": "stored"}
    except Exception as e:
        db.rollback()
        logger.exception("DB feedback failed: %s", e)
        raise HTTPException(status_code=500, detail="db error")
    finally:
        db.close()
