import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

DATA = Path("data/processed")
MODELS = Path("models")


def main(k=5):
    tr = pd.read_csv(DATA / "train.csv")
    X = tr["text"].astype(str).values
    y = tr["label"].values

    meta_path = MODELS / "metadata.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    model = joblib.load(meta["model_file"])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    ts = np.linspace(0.2, 0.8, 61)
    f1s = np.zeros_like(ts, dtype=float)

    for _, val_idx in skf.split(X, y):
        proba = model.predict_proba(X[val_idx])[:, 1]
        yv = y[val_idx]
        for i, t in enumerate(ts):
            pred = (proba >= t).astype(int)
            f1s[i] += f1_score(yv, pred, average="macro")

    f1s /= k
    best_i = int(np.argmax(f1s))
    best_t, best_f1 = float(ts[best_i]), float(f1s[best_i])
    print(f"[CV THRESHOLD] t={best_t:.3f} | mean macro-F1={best_f1:.4f}")

    meta["threshold"] = best_t
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[META UPDATED] threshold saved.")


if __name__ == "__main__":
    main()
