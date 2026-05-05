#!/usr/bin/env python3
"""P06 handcrafted-feature baselines for target signature classification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.target_signature import TARGET_CLASSES
from train import assert_current_schema


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    assert_current_schema(path)
    with h5py.File(path, "r") as f:
        return f["features"][:].astype(np.float32), f["y"][:].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--train_data_dir", default=None)
    ap.add_argument("--eval_data_dir", default=None)
    ap.add_argument("--eval_split", choices=["val", "test"], default="test")
    ap.add_argument("--model", choices=["logreg", "linear_svm", "rbf_svm"], default="logreg")
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/feature_baseline_results.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_data_dir = Path(args.train_data_dir) if args.train_data_dir else data_dir
    eval_data_dir = Path(args.eval_data_dir) if args.eval_data_dir else data_dir
    x_train, y_train = load_features(train_data_dir / "signature_train.h5")
    x_val, y_val = load_features(train_data_dir / "signature_val.h5")
    x_eval, y_eval = load_features(eval_data_dir / f"signature_{args.eval_split}.h5")

    if args.max_train and len(y_train) > args.max_train:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(y_train), size=args.max_train, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]

    if args.model == "linear_svm":
        clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", max_iter=10000))
    elif args.model == "rbf_svm":
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced"))
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))

    clf.fit(x_train, y_train)
    val_pred = clf.predict(x_val)
    eval_pred = clf.predict(x_eval)
    payload = {
        "kind": "p06_feature_baseline",
        "model": args.model,
        "class_names": list(TARGET_CLASSES),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "eval_samples": int(len(y_eval)),
        "eval_split": args.eval_split,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "eval_accuracy": float(accuracy_score(y_eval, eval_pred)),
        "eval_f1_macro": float(f1_score(y_eval, eval_pred, average="macro", zero_division=0)),
        "eval_confusion_matrix": confusion_matrix(y_eval, eval_pred, labels=list(range(len(TARGET_CLASSES)))).tolist(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
