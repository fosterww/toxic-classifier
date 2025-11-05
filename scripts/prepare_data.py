from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

try:
    from app.utils import clean_text
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.utils import clean_text

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15
DEFAULT_SEED = 42


def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split()).lower()
    return s


def _hash_group(s: str) -> str:
    return hashlib.sha1(_norm_text(s).encode("utf-8")).hexdigest()

def _load_jigsaw_if_exists(limit:int|None=20000) -> pd.DataFrame | None:
    candidates = [
        RAW / "jigsaw_train.csv",
        RAW / "train.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    df = pd.read_csv(path)
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=42)
    text_col = "comment_text" if "comment_text" in df.columns else "text"
    if text_col not in df.columns:
        raise ValueError(
            f"Не найден текстовый столбец в {path}. Ожидал 'comment_text' или 'text'."
        )

    if "toxic" in df.columns:
        out = df[[text_col, "toxic"]].rename(columns={text_col: "text", "toxic": "label"})
        out["label"] = (out["label"] >= 1).astype(int)
    else:
        multilabel_cols = [c for c in df.columns if c not in {text_col}]
        if not multilabel_cols:
            raise ValueError(
                f"В {path} нет столбца 'toxic' и нет мульти-классов для свертки."
            )
        out = df[[text_col] + multilabel_cols].rename(columns={text_col: "text"})
        out["label"] = (out[multilabel_cols].sum(axis=1) > 0).astype(int)
        out = out[["text", "label"]]

    out = out.dropna(subset=["text", "label"])
    out["text"] = out["text"].astype(str).str.strip()
    out["label"] = out["label"].astype(int)
    return out


def _load_extra_if_exists() -> pd.DataFrame | None:
    path = RAW / "extra_ru_ua.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    need = {"text", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} должен содержать колонки {need}")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


def load_sources() -> pd.DataFrame:
    dfs = []
    jigsaw = _load_jigsaw_if_exists(limit=20000)
    if jigsaw is not None and len(jigsaw):
        dfs.append(jigsaw)

    extra = _load_extra_if_exists()
    if extra is not None and len(extra):
        dfs.append(extra)

    if not dfs:
        raise FileNotFoundError(
            "Не нашли данных. Положи 'jigsaw_train.csv' или 'train.csv' и/или 'extra_ru_ua.csv' в data/raw/."
        )

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.dropna(subset=["text", "label"])
    all_df["text"] = all_df["text"].astype(str).str.strip()
    all_df = all_df[all_df["text"].str.len() > 0]
    all_df["label"] = all_df["label"].astype(int)
    return all_df[["text", "label"]]


def group_split(df, val_size: float, test_size: float, seed: int):
    if not 0 < val_size < 0.5 or not 0 < test_size < 0.5:
        raise ValueError("val_size и test_size должны быть в (0, 0.5).")

    df = df.copy()
    df["__group"] = df["text"].map(_hash_group)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["__group"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    temp_test_size = test_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=temp_test_size, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["__group"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    for part in (train_df, val_df, test_df):
        part.drop(columns=["__group"], inplace=True, errors="ignore")

    return train_df, val_df, test_df

def main(
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
):
    print("=== PREPARE DATA ===")
    print("RAW:", RAW); print("OUT:", OUT)
    df = load_sources()

    df["norm"] = df["text"].map(clean_text)
    before = len(df)
    df = df.drop_duplicates(subset=["norm"]).drop(columns=["norm"]).reset_index(drop=True)
    print(f"[DEDUP] exact duplicates removed: {before - len(df)} | remain: {len(df)}")

    y = df["label"].values
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(df, y))
    trainval, test_df = df.iloc[idx_trainval], df.iloc[idx_test]
    y_tv = trainval["label"].values
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed)
    idx_train, idx_val = next(sss2.split(trainval, y_tv))
    train_df, val_df = trainval.iloc[idx_train], trainval.iloc[idx_val]

    for name, d in [("ALL", df), ("train", train_df), ("val", val_df), ("test", test_df)]:
        pos = int((d["label"]==1).sum()); n=len(d)
        print(f"{name:>5}: n={n:4d} | pos={pos:4d} | neg={n-pos:4d} | pos_ratio={pos/n: .3f}")

    train_df.to_csv(OUT/"train.csv", index=False)
    val_df.to_csv(OUT/"val.csv", index=False)
    test_df.to_csv(OUT/"test.csv", index=False)
    print("Saved train/val/test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        main(val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(1)
