import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

try:
    from app.utils import clean_text
except Exception:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.utils import clean_text

DATA = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

PARAM_GRID = {
    "tfidf__min_df": [1, 2, 3],
    "tfidf__max_df": [0.9, 0.95],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.5, 1.0, 2.0],
}


def build_pipeline():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50_000)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1500,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")

    for df in (train, val):
        df["text"] = df["text"].astype(str).map(clean_text)

    pipe = build_pipeline()
    gs = GridSearchCV(
        pipe,
        PARAM_GRID,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(train["text"], train["label"])

    best = gs.best_estimator_
    best.fit(
        pd.concat([train["text"], val["text"]]),
        pd.concat([train["label"], val["label"]]),
    )

    val_pred = gs.best_estimator_.predict(val["text"])
    val_f1 = f1_score(val["label"], val_pred, average="macro")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = MODELS / f"model_{ts}.joblib"
    joblib.dump(best, model_path)

    meta = {
        "created_at": ts,
        "model_file": str(model_path),
        "features": "tfidf grid-tuned",
        "best_params": gs.best_params_,
        "clf": "logreg(class_weight=balanced)",
        "val_macro_f1": float(val_f1),
        "cv": 3,
        "notes": "clean_text applied; refit on train+val",
    }
    with open(MODELS / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    hist = MODELS / "history.csv"
    line = f'{ts},{val_f1:.5f},"{json.dumps(gs.best_params_)}"\n'
    if not hist.exists():
        hist.write_text("timestamp,val_macro_f1,best_params\n", encoding="utf-8")
    with open(hist, "a", encoding="utf-8") as h:
        h.write(line)

    print("[BEST PARAMS]", gs.best_params_)
    print(f"[VAL macro-F1] {val_f1:.4f}")
    print("[SAVED]", model_path)


if __name__ == "__main__":
    main()
