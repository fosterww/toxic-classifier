import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

DATA = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    train["text"] = train["text"].astype(str)
    val["text"] = val["text"].astype(str)
    return train, val


def build_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), max_features=100_000, lowercase=True
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    train_df, val_df = load_data()
    Xtr, ytr = train_df["text"], train_df["label"]
    Xv, yv = val_df["text"], val_df["label"]

    pipe = build_pipeline()

    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0

    ypred = pipe.predict(Xv)
    macro_f1 = f1_score(yv, ypred, average="macro")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = MODELS / f"model_{ts}.joblib"
    joblib.dump(pipe, model_path)

    meta = {
        "created": ts,
        "model_file": model_path.as_posix(),
        "features": "tfidf(1,2),max_features=100k",
        "clf": "logreg(class_weight=balanced,max_iter=1000)",
        "split": {"train": len(train_df), "val": len(val_df)},
        "val_macro_f1": float(macro_f1),
        "train_time_sec": round(train_time, 3),
        "random_state": 42,
    }
    with open(MODELS / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved model -> {model_path}")
    print(f"[METRIC] val macro-F1: {macro_f1:.4f} | train_time: {train_time:.2f}s")


if __name__ == "__main__":
    main()
