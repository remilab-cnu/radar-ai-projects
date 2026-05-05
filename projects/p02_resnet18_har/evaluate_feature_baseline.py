#!/usr/bin/env python3
"""P02 handcrafted-feature baseline for lecture comparisons.

The generator already stores micro-Doppler descriptor vectors in `features`.
This script trains a small classical classifier on those features so the lecture
can compare hand-designed radar descriptors with the ResNet spectrogram model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.micro_doppler import ACTIVITY_LABELS, N_CLASSES
from train import assert_current_schema


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    assert_current_schema(path)
    with h5py.File(path, "r") as f:
        return f["features"][:].astype(np.float32), f["y"][:].astype(np.int64)


def limit_train_samples(
    x: np.ndarray,
    y: np.ndarray,
    max_train: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally subsample training features for slower classical baselines."""
    if max_train is None or max_train <= 0 or len(y) <= max_train:
        return x, y
    rng = np.random.default_rng(seed)
    keep = []
    per_class = max(1, max_train // N_CLASSES)
    for cls in range(N_CLASSES):
        idx = np.where(y == cls)[0]
        n_keep = min(len(idx), per_class)
        keep.append(rng.choice(idx, size=n_keep, replace=False))
    keep = np.concatenate(keep)
    if len(keep) < max_train:
        remaining = np.setdiff1d(np.arange(len(y)), keep, assume_unique=False)
        extra = min(len(remaining), max_train - len(keep))
        if extra > 0:
            keep = np.concatenate([keep, rng.choice(remaining, size=extra, replace=False)])
    rng.shuffle(keep)
    return x[keep], y[keep]


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for idx, name in enumerate(ACTIVITY_LABELS):
        mask = y_true == idx
        if np.any(mask):
            out[name] = float(np.mean(y_pred[mask] == y_true[mask]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument(
        "--train_data_dir",
        default=None,
        help="feature training split directory; defaults to --data_dir",
    )
    ap.add_argument(
        "--eval_data_dir",
        default=None,
        help="feature evaluation split directory; defaults to --data_dir",
    )
    ap.add_argument(
        "--eval_split",
        choices=["val", "test"],
        default="test",
        help="split inside --eval_data_dir used for cross-dataset evaluation",
    )
    ap.add_argument("--model", choices=["logreg", "linear_svm", "rbf_svm"], default="logreg")
    ap.add_argument("--max_train", type=int, default=None,
                    help="optional stratified cap for training samples; useful for rbf_svm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/feature_baseline_results.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_data_dir = Path(args.train_data_dir) if args.train_data_dir else data_dir
    eval_data_dir = Path(args.eval_data_dir) if args.eval_data_dir else data_dir

    x_train, y_train = load_features(train_data_dir / "har_train.h5")
    x_val, y_val = load_features(train_data_dir / "har_val.h5")
    x_eval, y_eval = load_features(eval_data_dir / f"har_{args.eval_split}.h5")
    original_train_samples = int(len(y_train))
    x_train, y_train = limit_train_samples(x_train, y_train, args.max_train, args.seed)

    if args.model == "linear_svm":
        clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", max_iter=10000))
    elif args.model == "rbf_svm":
        # Matches the earlier lecture-material baseline family.  RBF SVC can be
        # quadratic in sample count, so --max_train is recommended for quick
        # classroom runs on the full 30k split.
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
        )
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
        )

    clf.fit(x_train, y_train)
    val_pred = clf.predict(x_val)
    eval_pred = clf.predict(x_eval)
    eval_accuracy = float(accuracy_score(y_eval, eval_pred))
    eval_per_class = per_class_accuracy(y_eval, eval_pred)
    eval_confusion = confusion_matrix(y_eval, eval_pred, labels=list(range(N_CLASSES))).tolist()

    payload = {
        "kind": "p02_feature_baseline",
        "model": args.model,
        "train_data_dir": str(train_data_dir),
        "eval_data_dir": str(eval_data_dir),
        "eval_split": args.eval_split,
        "class_names": list(ACTIVITY_LABELS),
        "n_classes": int(N_CLASSES),
        "train_samples": int(len(y_train)),
        "original_train_samples": original_train_samples,
        "max_train": args.max_train,
        "val_samples": int(len(y_val)),
        "eval_samples": int(len(y_eval)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "eval_accuracy": eval_accuracy,
        "eval_per_class": eval_per_class,
        "eval_confusion_matrix": eval_confusion,
    }
    if args.eval_split == "test":
        payload.update({
            "test_samples": int(len(y_eval)),
            "test_accuracy": eval_accuracy,
            "test_per_class": eval_per_class,
            "test_confusion_matrix": eval_confusion,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
