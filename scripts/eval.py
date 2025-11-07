import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA = Path("data/processed")
MODELS = Path("models")
ART = Path("notebooks")


def load_latest_model():
    meta = json.load(open(MODELS / "metadata.json", encoding="utf-8"))
    model = joblib.load(meta["model_file"])
    return model, meta


def plot_confusion(cm: np.ndarray, out_path: Path, labels=("clean", "toxic")):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(split="test"):
    df = pd.read_csv(DATA / f"{split}.csv")
    df["text"] = df["text"].astype(str)

    model, meta = load_latest_model()

    y_true = df["label"].values
    y_pred = model.predict(df["text"])
    print("[META]", json.dumps(meta, indent=2, ensure_ascii=False))
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    out_png = ART / f"confusion_{split}.png"
    plot_confusion(cm, out_png)
    print(f"[ASSET] Saved confusion matrix -> {out_png}")

    try:
        y_proba = model.predict_proba(df["text"])[:, 1]
        roc_auc = roc_auc_score(y_true, y_proba)
        print(f"[METRIC] ROC-AUC ({split}): {roc_auc:.4f}")
    except Exception as e:
        print("[WARN] ROC-AUC not available:", e)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="test")
    args = p.parse_args()
    main(split=args.split)
