#!/usr/bin/env python3
"""Generate deterministic P04 qualitative case studies from eval diagnostics.

This script is intentionally P04-local and reads the diagnostic artifacts emitted
by ``train.py --eval_only``:

    artifacts/eval_results.json
    artifacts/per_sample_metrics.csv

It selects examples by a fixed anti-cherry-picking taxonomy and renders PNGs
under ``artifacts/case_studies/``.  The figures are evidence artifacts, not the
lecture page itself; lecture figure numbers and narrative belong in HTML.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import DnCNNSAR, frost_filter, lee_filter, median_filter


def _float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _finite_rows(rows: list[dict], key: str, *, source: str | None = None) -> list[dict]:
    out = []
    for row in rows:
        if source and row.get("source", "").lower() != source.lower():
            continue
        value = _float(row, key)
        if np.isfinite(value):
            out.append(row)
    return out


def _pick_unique(
    name: str,
    candidates: list[dict],
    used: set[int],
    unavailable: dict,
) -> dict | None:
    for row in candidates:
        idx = int(row["sample_index"])
        if idx not in used:
            used.add(idx)
            return row
    unavailable[name] = "no unused candidate matched selection rule"
    return None


def select_cases(rows: list[dict]) -> tuple[list[dict], dict]:
    """Select deterministic best/median/worst/failure examples."""
    selected: list[dict] = []
    unavailable: dict[str, str] = {}
    used: set[int] = set()

    slc = _finite_rows(rows, "dncnn_raw_psnr", source="slc")
    grd = _finite_rows(rows, "dncnn_raw_psnr", source="grd")

    def add(case_name: str, reason: str, candidates: list[dict]) -> None:
        row = _pick_unique(case_name, candidates, used, unavailable)
        if row is None:
            return
        case = dict(row)
        case["case_name"] = case_name
        case["selection_reason"] = reason
        selected.append(case)

    if slc:
        psnrs = np.array([_float(r, "dncnn_raw_psnr") for r in slc])
        ssims = np.array([_float(r, "dncnn_raw_ssim") for r in slc])
        med_psnr = float(np.median(psnrs))
        med_ssim = float(np.median(ssims))
        psnr_scale = float(np.std(psnrs)) or 1.0
        ssim_scale = float(np.std(ssims)) or 1.0
        add(
            "slc_median",
            "SLC sample closest to median raw DnCNN PSNR and SSIM.",
            sorted(
                slc,
                key=lambda r: abs(_float(r, "dncnn_raw_psnr") - med_psnr) / psnr_scale
                + abs(_float(r, "dncnn_raw_ssim") - med_ssim) / ssim_scale,
            ),
        )
        add(
            "slc_best",
            "Highest finite SLC raw DnCNN PSNR.",
            sorted(slc, key=lambda r: _float(r, "dncnn_raw_psnr"), reverse=True),
        )
        add(
            "slc_worst",
            "Lowest finite SLC raw DnCNN PSNR.",
            sorted(slc, key=lambda r: _float(r, "dncnn_raw_psnr")),
        )
    else:
        unavailable["slc_median"] = "no SLC rows with finite raw DnCNN PSNR"
        unavailable["slc_best"] = "no SLC rows with finite raw DnCNN PSNR"
        unavailable["slc_worst"] = "no SLC rows with finite raw DnCNN PSNR"

    add(
        "out_of_range",
        "Largest raw DnCNN out-of-range / clipping fraction.",
        sorted(
            rows,
            key=lambda r: (
                _float(r, "dncnn_raw_clipped_fraction"),
                max(abs(min(_float(r, "dncnn_raw_min"), 0.0)), max(_float(r, "dncnn_raw_max") - 1.0, 0.0)),
            ),
            reverse=True,
        ),
    )

    baseline_win_rows = [
        r for r in rows
        if np.isfinite(_float(r, "dncnn_raw_vs_best_classical_psnr_delta"))
        and _float(r, "dncnn_raw_vs_best_classical_psnr_delta") < 0
    ]
    if baseline_win_rows:
        add(
            "baseline_wins",
            "Most negative raw DnCNN PSNR delta against the best classical baseline.",
            sorted(baseline_win_rows, key=lambda r: _float(r, "dncnn_raw_vs_best_classical_psnr_delta")),
        )
    else:
        unavailable["baseline_wins"] = "no evaluated sample where a classical baseline beats raw DnCNN on PSNR"

    add(
        "oversmooth_candidate",
        "High DnCNN ENL log-domain proxy with below-median SSIM when possible.",
        sorted(
            rows,
            key=lambda r: (
                "high_enl_proxy_oversmooth_candidate" in r.get("failure_flags", ""),
                _float(r, "dncnn_raw_enl_log_roi_proxy"),
            ),
            reverse=True,
        ),
    )

    if slc:
        # Metric disagreement: high SSIM despite weak PSNR, or vice versa.  This
        # exposes why a single scalar is insufficient for SAR despeckling.
        add(
            "metric_disagreement",
            "SLC case with high SSIM rank but weak PSNR rank.",
            sorted(
                slc,
                key=lambda r: _float(r, "dncnn_raw_ssim") - 0.03 * _float(r, "dncnn_raw_psnr"),
                reverse=True,
            ),
        )
    else:
        unavailable["metric_disagreement"] = "no SLC rows available"

    if grd:
        add(
            "grd_easy",
            "GRD minority/easier prefiltered case with high raw DnCNN PSNR.",
            sorted(grd, key=lambda r: _float(r, "dncnn_raw_psnr"), reverse=True),
        )
    else:
        unavailable["grd_easy"] = "no GRD rows available"

    return selected, unavailable


def _load_patch(data_path: Path, sample_index: int) -> tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(data_path, "r") as f:
        noisy = f["noisy"][sample_index, 0].astype("float32")
        clean = f["clean"][sample_index, 0].astype("float32")
        source = f["source"][sample_index]
        if isinstance(source, bytes):
            source = source.decode("utf-8")
    return noisy, clean, str(source).upper()


def _model_output(model: torch.nn.Module, noisy: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        inp = torch.from_numpy(noisy[np.newaxis, np.newaxis]).float()
        return model(inp).numpy()[0, 0]


def _case_metrics(row: dict) -> dict:
    keys = [
        "dncnn_raw_psnr", "dncnn_raw_ssim", "dncnn_clipped_psnr", "dncnn_clipped_ssim",
        "best_classical_psnr_method", "best_classical_psnr",
        "dncnn_raw_vs_best_classical_psnr_delta",
        "dncnn_raw_min", "dncnn_raw_max", "dncnn_raw_clipped_fraction",
        "failure_flags",
    ]
    return {k: row.get(k, "") for k in keys}


def render_case(case: dict, model: torch.nn.Module, data_path: Path, out_dir: Path) -> dict:
    sample_index = int(case["sample_index"])
    noisy, clean, source = _load_patch(data_path, sample_index)
    pred_raw = _model_output(model, noisy)
    pred_clipped = np.clip(pred_raw, 0.0, 1.0)
    lee = lee_filter(noisy, window_size=7)
    frost = frost_filter(noisy, window_size=7)
    median = median_filter(noisy, window_size=7)
    err = np.abs(pred_clipped - clean)

    panels = [
        ("Noisy", noisy, "gray"),
        ("Pseudo-clean", clean, "gray"),
        ("DnCNN raw", pred_raw, "gray"),
        ("DnCNN clipped", pred_clipped, "gray"),
        ("Lee", lee, "gray"),
        ("Frost", frost, "gray"),
        ("Median", median, "gray"),
        ("|DnCNN clip - target|", err, "magma"),
    ]

    display_stack = np.stack([noisy, clean])
    vmin = float(np.percentile(display_stack, 1))
    vmax = float(np.percentile(display_stack, 99))
    err_vmax = float(max(np.percentile(err, 99), 1e-3))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        if cmap == "magma":
            im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=err_vmax)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    filename = f"{case['case_name']}_idx{sample_index:04d}_{source.lower()}.png"
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "case_name": case["case_name"],
        "selection_reason": case["selection_reason"],
        "sample_index": sample_index,
        "source": source,
        "figure": str(path),
        "display_vmin": vmin,
        "display_vmax": vmax,
        "error_vmax": err_vmax,
        "metrics": _case_metrics(case),
        "caveat": "Pseudo-clean Sentinel-1 target; not true clean SAR ground truth.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic P04 diagnostic case-study figures")
    root = Path(__file__).resolve().parent
    parser.add_argument("--data_dir", type=Path, default=root / "data")
    parser.add_argument("--artifact_dir", type=Path, default=root / "artifacts")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metrics_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--n_filters", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=17)
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.artifact_dir / "best_model.pt"
    metrics_csv = args.metrics_csv or args.artifact_dir / "per_sample_metrics.csv"
    out_dir = args.out_dir or args.artifact_dir / "case_studies"
    data_path = args.data_dir / "real_despeckling_test.h5"

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not metrics_csv.exists():
        raise FileNotFoundError(f"per-sample metrics CSV not found: {metrics_csv}")
    if not data_path.exists():
        raise FileNotFoundError(f"test HDF5 not found: {data_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(metrics_csv)
    selected, unavailable = select_cases(rows)

    model = DnCNNSAR(n_channels=1, n_filters=args.n_filters, n_layers=args.n_layers)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    cases = [render_case(case, model, data_path, out_dir) for case in selected]
    manifest = {
        "checkpoint": str(checkpoint),
        "data_file": str(data_path),
        "metrics_csv": str(metrics_csv),
        "model": {"n_filters": args.n_filters, "n_layers": args.n_layers},
        "selection_policy": [
            "slc_median", "slc_best", "slc_worst", "out_of_range",
            "baseline_wins", "oversmooth_candidate", "metric_disagreement", "grd_easy",
        ],
        "unavailable_cases": unavailable,
        "cases": cases,
    }
    manifest_path = out_dir / "case_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    readme = out_dir / "README.md"
    with readme.open("w") as f:
        f.write("# P04 Diagnostic Case Studies\n\n")
        f.write("Generated deterministically from `per_sample_metrics.csv`.\n\n")
        f.write("Caveat: targets are pseudo-clean Sentinel-1 multi-look products, not true clean SAR ground truth.\n\n")
        for case in cases:
            f.write(f"- **{case['case_name']}** idx `{case['sample_index']}` ({case['source']}): {case['selection_reason']} → `{Path(case['figure']).name}`\n")
        if unavailable:
            f.write("\n## Unavailable case classes\n\n")
            for name, reason in unavailable.items():
                f.write(f"- `{name}`: {reason}\n")

    print(f"Saved {len(cases)} case figures to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
