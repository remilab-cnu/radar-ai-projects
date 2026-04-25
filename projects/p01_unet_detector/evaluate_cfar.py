#!/usr/bin/env python3
"""Run verified CA-CFAR sweeps for P01 from linear RDM data."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils import (
    add_counts,
    choose_by_max_f1,
    confusion_counts,
    iter_samples,
    load_policy,
    metrics_from_counts,
    split_path,
)
from shared.fmcw_simulator import ca_cfar_2d


def parse_pair(text: str) -> tuple[int, int]:
    a, b = text.split(",")
    return int(a), int(b)


def run_policy(
    path: Path,
    guard: tuple[int, int],
    train: tuple[int, int],
    pfa: float,
    max_samples: int | None,
) -> dict:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    n = 0
    for sample in iter_samples(path, max_samples):
        det = ca_cfar_2d(sample["rdm_mag_linear"], guard=guard, train=train, pfa=pfa)
        add_counts(counts, confusion_counts(det, sample["gt"]))
        n += 1
    metrics = metrics_from_counts(counts)
    metrics.update({
        "guard": list(guard),
        "train": list(train),
        "pfa_design": float(pfa),
        "n_eval": n,
        "false_alarms_per_rdm": float(counts["fp"] / max(n, 1)),
    })
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--policy_from", "--policy-from", dest="policy_from")
    ap.add_argument("--guard", default="2,2")
    ap.add_argument("--train", default="4,4")
    ap.add_argument("--pfa", type=float, default=1e-4)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    path = split_path(args.data_dir, args.split)
    if args.policy_from:
        pol = load_policy(args.policy_from)
        guards = [tuple(pol["guard"])]
        trains = [tuple(pol["train"])]
        pfas = [float(pol.get("pfa_design", pol.get("pfa", args.pfa)))]
    elif args.sweep:
        guards = [(1, 1), (2, 2), (3, 3)]
        trains = [(4, 4), (6, 6), (8, 8), (10, 10)]
        pfas = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    else:
        guards = [parse_pair(args.guard)]
        trains = [parse_pair(args.train)]
        pfas = [args.pfa]

    results = [run_policy(path, g, tr, pfa, args.max_samples) for g, tr, pfa in product(guards, trains, pfas)]
    selected = results[0] if args.policy_from or not args.sweep else choose_by_max_f1(results)
    payload = {
        "kind": "p01_cfar",
        "split": args.split,
        "data_path": str(path),
        "selected_policy": selected,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(selected, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
