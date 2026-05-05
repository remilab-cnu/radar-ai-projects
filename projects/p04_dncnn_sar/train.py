#!/usr/bin/env python3
"""Train/evaluate P04 DnCNN-SAR on real Sentinel-1 despeckling patches.

This is the primary classroom implementation for P04.  It consumes HDF5 files
created by ``generate_data.py`` from real Sentinel-1 GRD/SLC products:

    data/real_despeckling_train.h5
    data/real_despeckling_val.h5
    data/real_despeckling_test.h5

Examples
--------
    # Full real-data training, generating patches first
    python train.py --generate --epochs 100 --batch_size 32 --lr 5e-4 --no_amp

    # Fast CPU smoke check: GRD-only data generation + tiny model
    python train.py --generate --smoke

    # Evaluation only
    python train.py --eval_only --checkpoint artifacts/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from model import (
    DnCNNSAR,
    DespecklingLoss,
    compute_enl,
    compute_epi,
    compute_psnr,
    compute_ssim,
    count_parameters,
    frost_filter,
    lee_filter,
    median_filter,
)


METRIC_KEYS = ("psnr", "ssim", "enl_log_roi_proxy", "epi")


class RealDespecklingDataset(Dataset):
    """HDF5 dataset for real Sentinel-1 SAR despeckling data.

    Expected keys:
        noisy  : (N, 1, H, W) float32, speckled SAR log-dB in [0, 1]
        clean  : (N, 1, H, W) float32, pseudo-clean SAR log-dB in [0, 1]
        source : (N,) bytes/str, ``grd`` or ``slc``
    """

    def __init__(self, h5_path: str | Path, augment: bool = False) -> None:
        self.h5_path = Path(h5_path)
        self.augment = augment
        with h5py.File(self.h5_path, "r") as f:
            self.n_samples = int(f["noisy"].shape[0])
            self.image_size = int(f["noisy"].shape[2])
        self._file = None

    def _open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        self._open()
        noisy = self._file["noisy"][idx].copy()
        clean = self._file["clean"][idx].copy()

        if self.augment:
            # Preserve paired transforms for noisy/clean patches.
            if np.random.random() < 0.5:
                noisy = noisy[:, :, ::-1].copy()
                clean = clean[:, :, ::-1].copy()
            k = np.random.randint(4)
            if k > 0:
                noisy = np.rot90(noisy[0], k=k).copy()[np.newaxis, :, :]
                clean = np.rot90(clean[0], k=k).copy()[np.newaxis, :, :]

        return torch.from_numpy(noisy), torch.from_numpy(clean)

    def get_meta(self, idx: int) -> dict[str, str]:
        self._open()
        source = self._file["source"][idx]
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        return {"source": str(source)}

    @property
    def is_real(self) -> bool:
        return True


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred = model(noisy)
            loss = criterion(pred, clean)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_psnr(model, loader, criterion, device, use_amp=False):
    """Return (mean_loss, mean_psnr) over the validation loader."""
    model.eval()
    total_loss = 0.0
    psnr_sum = 0.0
    n_samples = 0
    n_batches = 0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred = model(noisy)
            loss = criterion(pred, clean)

        pred_np = pred.cpu().float().numpy()
        clean_np = clean.cpu().float().numpy()
        for b in range(pred_np.shape[0]):
            psnr_sum += compute_psnr(pred_np[b, 0], clean_np[b, 0])
        n_samples += pred_np.shape[0]
        total_loss += float(loss.item())
        n_batches += 1

    mean_psnr = psnr_sum / n_samples if n_samples > 0 else 0.0
    return total_loss / max(n_batches, 1), mean_psnr


def _json_safe(value):
    """Convert numpy scalars / non-finite floats into JSON-safe values."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _h5_attrs(path: Path) -> dict:
    """Return HDF5 attrs in JSON-safe form."""
    with h5py.File(path, "r") as f:
        return {k: _json_safe(v) for k, v in f.attrs.items()}


def _source_counts(dataset, indices: list[int] | None = None) -> dict[str, int]:
    """Count GRD/SLC labels for either the whole dataset or selected indices."""
    if indices is None:
        indices = list(range(len(dataset)))
    counts: dict[str, int] = {}
    for idx in indices:
        source = dataset.get_meta(idx)["source"].upper()
        counts[source] = counts.get(source, 0) + 1
    return counts


def _eval_indices(dataset_len: int, eval_samples: int) -> list[int]:
    """Resolve eval sample count.  A non-positive value means the full test set."""
    if eval_samples <= 0:
        return list(range(dataset_len))
    return list(range(min(dataset_len, eval_samples)))


def _edge_corr_details(filtered: np.ndarray, original: np.ndarray, reference: np.ndarray) -> dict:
    """Return EPI plus numerator/denominator diagnostics.

    The public ``compute_epi`` returns only the ratio.  For diagnostics we keep the
    correlation terms so unstable denominator cases can be identified.
    """
    from scipy.ndimage import laplace

    def edge(x: np.ndarray) -> np.ndarray:
        return np.abs(laplace(x.astype(np.float64)))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        denom = float(np.std(a_flat) * np.std(b_flat))
        if denom < 1e-10:
            return 0.0
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    e_filt = edge(filtered)
    e_orig = edge(original)
    e_ref = edge(reference)
    numerator = corr(e_filt, e_ref)
    denominator = corr(e_orig, e_ref)
    valid = bool(abs(denominator) >= 1e-10)
    epi = float(numerator / denominator) if valid else 0.0
    return {
        "epi": epi,
        "epi_numerator": numerator,
        "epi_denominator": denominator,
        "epi_denominator_abs": abs(denominator),
        "epi_valid": valid,
    }


def _compute_metrics(output: np.ndarray, noisy: np.ndarray, clean: np.ndarray, roi: tuple) -> dict:
    """Compute metrics in the normalized log/dB evaluation domain."""
    epi = _edge_corr_details(output, noisy, clean)
    return {
        "psnr": compute_psnr(output, clean),
        "ssim": compute_ssim(output, clean),
        # Current images are normalized log/dB; this is a smoothness proxy, not
        # physical linear-intensity ENL.
        "enl_log_roi_proxy": compute_enl(output, roi=roi),
        **epi,
    }


def _range_stats(img: np.ndarray) -> dict:
    """Output-range diagnostics for normalized images."""
    vals = img.astype(np.float64).ravel()
    below = vals < 0.0
    above = vals > 1.0
    return {
        "min": float(np.min(vals)),
        "p001": float(np.percentile(vals, 0.1)),
        "p01": float(np.percentile(vals, 1.0)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p99": float(np.percentile(vals, 99.0)),
        "p999": float(np.percentile(vals, 99.9)),
        "max": float(np.max(vals)),
        "below0_count": int(np.sum(below)),
        "above1_count": int(np.sum(above)),
        "clipped_count": int(np.sum(below | above)),
        "below0_fraction": float(np.mean(below)),
        "above1_fraction": float(np.mean(above)),
        "clipped_fraction": float(np.mean(below | above)),
        "is_finite": bool(np.all(np.isfinite(vals))),
    }


def _robust_stats(values: list[float]) -> dict:
    """Robust JSON-safe stats for skewed SAR metrics."""
    arr = np.array(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    out = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "invalid_count": int(arr.size - finite.size),
    }
    if finite.size == 0:
        out.update({
            "mean": None, "std": None, "median": None, "iqr": None,
            "p05": None, "p95": None, "min": None, "max": None,
        })
        return out
    q25, q75 = np.percentile(finite, [25, 75])
    out.update({
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "iqr": float(q75 - q25),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    })
    return out


def _aggregate_records(rows: list[dict], methods: list[str]) -> dict:
    """Aggregate wide per-sample rows into method metric summaries."""
    summary: dict[str, dict] = {}
    for method in methods:
        method_summary = {}
        for metric in METRIC_KEYS:
            stats = _robust_stats([r[f"{method}_{metric}"] for r in rows])
            method_summary[metric] = stats
            # Backward-friendly flat keys for the most common table fields.
            method_summary[f"{metric}_mean"] = stats["mean"]
            method_summary[f"{metric}_std"] = stats["std"]
            method_summary[f"{metric}_median"] = stats["median"]
            method_summary[f"{metric}_p05"] = stats["p05"]
            method_summary[f"{metric}_p95"] = stats["p95"]
        summary[method] = method_summary
    return summary


def _flag_rows(rows: list[dict]) -> None:
    """Add deterministic failure/selection helper flags to each per-sample row."""
    psnr_vals = np.array([r["dncnn_raw_psnr"] for r in rows], dtype=np.float64)
    psnr_finite = psnr_vals[np.isfinite(psnr_vals)]
    low_psnr = float(np.percentile(psnr_finite, 5)) if psnr_finite.size else -float("inf")

    enl_vals = np.array([r["dncnn_raw_enl_log_roi_proxy"] for r in rows], dtype=np.float64)
    enl_finite = enl_vals[np.isfinite(enl_vals)]
    high_enl = float(np.percentile(enl_finite, 95)) if enl_finite.size else float("inf")

    ssim_vals = np.array([r["dncnn_raw_ssim"] for r in rows], dtype=np.float64)
    ssim_finite = ssim_vals[np.isfinite(ssim_vals)]
    med_ssim = float(np.median(ssim_finite)) if ssim_finite.size else 0.0

    epi_abs = np.array([abs(r["dncnn_raw_epi"]) for r in rows], dtype=np.float64)
    epi_abs_finite = epi_abs[np.isfinite(epi_abs)]
    high_epi_abs = float(np.percentile(epi_abs_finite, 95)) if epi_abs_finite.size else float("inf")

    for row in rows:
        best_method, best_psnr = max(
            ((m, row[f"{m}_psnr"]) for m in ["lee", "frost", "median"]),
            key=lambda item: item[1],
        )
        row["best_classical_psnr_method"] = best_method
        row["best_classical_psnr"] = best_psnr
        row["dncnn_raw_vs_best_classical_psnr_delta"] = row["dncnn_raw_psnr"] - best_psnr
        row["dncnn_psnr_clip_delta"] = row["dncnn_clipped_psnr"] - row["dncnn_raw_psnr"]
        row["dncnn_ssim_clip_delta"] = row["dncnn_clipped_ssim"] - row["dncnn_raw_ssim"]

        flags = ["grd_easy_case" if row["source"] == "grd" else "slc_case"]
        if row["dncnn_raw_clipped_fraction"] > 0:
            flags.append("out_of_range")
        if abs(row["dncnn_psnr_clip_delta"]) > 0.5 or abs(row["dncnn_ssim_clip_delta"]) > 0.01:
            flags.append("high_clip_delta")
        if row["dncnn_raw_vs_best_classical_psnr_delta"] < 0:
            flags.append("classical_beats_dncnn")
        if row["dncnn_raw_psnr"] <= low_psnr:
            flags.append("low_dncnn_psnr")
        if row["dncnn_raw_enl_log_roi_proxy"] >= high_enl and row["dncnn_raw_ssim"] < med_ssim:
            flags.append("high_enl_proxy_oversmooth_candidate")
        if (not row["dncnn_raw_epi_valid"]) or abs(row["dncnn_raw_epi"]) >= high_epi_abs:
            flags.append("epi_unstable")
        row["failure_flags"] = ";".join(flags)


def _write_per_sample_csv(rows: list[dict], path: Path) -> None:
    """Write wide per-sample metrics for deterministic case selection."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k, "")) for k in fieldnames})
    print(f"  Per-sample metrics saved to {path}")


@torch.no_grad()
def evaluate_full(
    model,
    dataset,
    device,
    *,
    eval_samples: int = 0,
    checkpoint_path: str | Path | None = None,
    data_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    config: dict | None = None,
):
    """Compare DnCNN-SAR against Lee, Frost, Median, and diagnostic variants."""
    model.eval()
    indices = _eval_indices(len(dataset), eval_samples)
    n_eval = len(indices)
    capped = bool(eval_samples > 0 and eval_samples < len(dataset))
    print(f"\n  Evaluating {n_eval} / {len(dataset)} samples ({'capped' if capped else 'full'})...")

    methods = ["noisy", "dncnn_raw", "dncnn_clipped", "lee", "frost", "median"]
    rows: list[dict] = []

    for eval_pos, sample_idx in enumerate(indices, start=1):
        noisy_t, clean_t = dataset[sample_idx]
        source = dataset.get_meta(sample_idx)["source"].lower()

        noisy_np = noisy_t[0].numpy()
        clean_np = clean_t[0].numpy()

        inp = noisy_t.unsqueeze(0).to(device)
        pred_t = model(inp)
        pred_raw = pred_t.cpu().float().numpy()[0, 0]
        pred_clipped = np.clip(pred_raw, 0.0, 1.0)

        # Classical filters operate on the same normalized log/dB representation
        # as DnCNN.  They are same-input-contract baselines, not optimized
        # physical linear-intensity SAR filters.
        outputs = {
            "noisy": noisy_np,
            "dncnn_raw": pred_raw,
            "dncnn_clipped": pred_clipped,
            "lee": lee_filter(noisy_np, window_size=7),
            "frost": frost_filter(noisy_np, window_size=7),
            "median": median_filter(noisy_np, window_size=7),
        }

        h, w = noisy_np.shape
        enl_roi = (h // 2 - 32, h // 2 + 32, w // 2 - 32, w // 2 + 32)
        row: dict = {
            "sample_index": sample_idx,
            "eval_position": eval_pos - 1,
            "source": source,
            "n_eval": n_eval,
            "n_test_total": len(dataset),
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "data_file": str(data_path or getattr(dataset, "h5_path", "")),
        }

        for method, out in outputs.items():
            metrics = _compute_metrics(out, noisy_np, clean_np, enl_roi)
            for metric_key in METRIC_KEYS:
                row[f"{method}_{metric_key}"] = metrics[metric_key]
            if method.startswith("dncnn"):
                row[f"{method}_epi_numerator"] = metrics["epi_numerator"]
                row[f"{method}_epi_denominator"] = metrics["epi_denominator"]
                row[f"{method}_epi_denominator_abs"] = metrics["epi_denominator_abs"]
                row[f"{method}_epi_valid"] = metrics["epi_valid"]

        for prefix, stats in [
            ("dncnn_raw", _range_stats(pred_raw)),
            ("dncnn_clipped", _range_stats(pred_clipped)),
        ]:
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value

        rows.append(row)

        if eval_pos % 500 == 0 or eval_pos == n_eval:
            print(f"    [{eval_pos}/{n_eval}]")

    _flag_rows(rows)

    overall = _aggregate_records(rows, methods)
    by_source = {}
    for label in ["GRD", "SLC"]:
        source_key = label.lower()
        source_rows = [r for r in rows if r["source"] == source_key]
        if not source_rows:
            continue
        by_source[label] = {
            "n": len(source_rows),
            **_aggregate_records(source_rows, methods),
        }

    range_diag = {
        "dncnn_raw": {
            "clipped_fraction": _robust_stats([r["dncnn_raw_clipped_fraction"] for r in rows]),
            "below0_fraction": _robust_stats([r["dncnn_raw_below0_fraction"] for r in rows]),
            "above1_fraction": _robust_stats([r["dncnn_raw_above1_fraction"] for r in rows]),
            "min": _robust_stats([r["dncnn_raw_min"] for r in rows]),
            "max": _robust_stats([r["dncnn_raw_max"] for r in rows]),
        },
        "dncnn_clipped": {
            "psnr_delta_vs_raw": _robust_stats([r["dncnn_psnr_clip_delta"] for r in rows]),
            "ssim_delta_vs_raw": _robust_stats([r["dncnn_ssim_clip_delta"] for r in rows]),
        },
    }

    eval_meta = {
        "n_eval": n_eval,
        "n_test_total": len(dataset),
        "capped": capped,
        "eval_sample_limit": eval_samples,
        "evaluated_indices_policy": "first_n" if capped else "all",
        "evaluated_indices_first": indices[:10],
        "evaluated_indices_last": indices[-10:],
        "source_counts_total": _source_counts(dataset),
        "source_counts_evaluated": _source_counts(dataset, indices),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "data_file": str(data_path or getattr(dataset, "h5_path", "")),
        "data_attrs": _h5_attrs(Path(data_path)) if data_path else {},
        "config": config or {},
        "metric_notes": {
            "dncnn_raw": "Primary DnCNN output before post-processing.",
            "dncnn_clipped": "Diagnostic/display variant clipped to [0, 1]; do not silently replace raw metrics.",
            "enl_log_roi_proxy": "Smoothness proxy computed on normalized log/dB center ROI, not physical linear-intensity ENL.",
            "epi": "High-variance ratio metric; prefer robust stats/percentiles for interpretation.",
            "targets": "Pseudo-clean Sentinel-1 multi-look targets, not true clean SAR ground truth.",
            "classical_baselines": "Lee/Frost/Median are applied in the same normalized log/dB domain as DnCNN.",
        },
    }

    if output_csv:
        _write_per_sample_csv(rows, Path(output_csv))

    return {
        "eval_metadata": _json_safe(eval_meta),
        "overall": _json_safe(overall),
        "by_source": _json_safe(by_source),
        "output_range_diagnostics": _json_safe(range_diag),
    }


def _print_summary(summary: dict) -> None:
    overall = summary["overall"]
    methods_display = [
        ("noisy", "Noisy"),
        ("dncnn_raw", "DnCNN raw"),
        ("dncnn_clipped", "DnCNN clip"),
        ("lee", "Lee"),
        ("frost", "Frost"),
        ("median", "Median"),
    ]

    meta = summary.get("eval_metadata", {})
    if meta:
        print(
            f"\n  Eval scope: {meta.get('n_eval')} / {meta.get('n_test_total')} "
            f"({'capped' if meta.get('capped') else 'full'})"
        )

    print("\n" + "=" * 74)
    print(f"  {'Method':<12}  {'PSNR':>7}  {'SSIM':>7}  {'ENL-proxy':>10}  {'EPI med':>9}  {'EPI mean':>9}")
    print("-" * 74)
    for key, label in methods_display:
        if key not in overall:
            continue
        s = overall[key]
        print(
            f"  {label:<12}  "
            f"{s['psnr_mean']:>7.2f}  "
            f"{s['ssim_mean']:>7.3f}  "
            f"{s['enl_log_roi_proxy_mean']:>10.2f}  "
            f"{s['epi_median']:>9.3f}  "
            f"{s['epi_mean']:>9.3f}"
        )
    print("=" * 74)

    by_source = summary.get("by_source", {})
    if by_source:
        print("\n  --- Per-source breakdown (PSNR / SSIM) ---")
        header = f"  {'Source':<8}  {'N':>5}"
        for _, label in methods_display:
            header += f"  {label + ' PSNR':>10}  {label + ' SSIM':>9}"
        print(header)
        for label, data in by_source.items():
            row = f"  {label:<8}  {data['n']:>5d}"
            for key, _ in methods_display:
                s = data.get(key, {})
                row += f"  {s.get('psnr_mean', float('nan')):>10.2f}  {s.get('ssim_mean', float('nan')):>9.3f}"
            print(row)
    range_diag = summary.get("output_range_diagnostics", {}).get("dncnn_raw", {})
    if range_diag:
        clipped = range_diag["clipped_fraction"]
        print(
            "\n  DnCNN raw output range diagnostics: "
            f"clipped_fraction mean={clipped['mean']:.6f}, "
            f"p95={clipped['p95']:.6f}, max={clipped['max']:.6f}"
        )
    print()


def _save_summary(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {path}")


def _generate_real_data(args) -> None:
    """Invoke generate_data.py with the selected real-data options."""
    gen_path = Path(__file__).with_name("generate_data.py")
    spec = importlib.util.spec_from_file_location("p04_generate_data", gen_path)
    gen_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gen_mod)

    gen_args = [
        "--out_dir", str(args.data_dir),
        "--seed", str(args.seed),
        "--patch_size", str(args.patch_size),
        "--look_size", str(args.look_size),
        "--smooth_method", args.smooth_method,
        "--data_root", args.data_root,
    ]
    if args.grd_path:
        gen_args.extend(["--grd_path", args.grd_path])
    for slc_dir in args.slc_dir or []:
        gen_args.extend(["--slc_dir", slc_dir])
    if args.smoke:
        gen_args.append("--smoke")

    old_argv = sys.argv
    try:
        sys.argv = ["generate_data.py"] + gen_args
        gen_mod.main()
    finally:
        sys.argv = old_argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P04: DnCNN-SAR on real Sentinel-1 despeckling data")
    parser.add_argument("--generate", action="store_true", help="Generate real Sentinel-1 HDF5 data before training")
    parser.add_argument("--generate_real", action="store_true", help="Alias for --generate")
    parser.add_argument("--real_data", action="store_true", help="Compatibility no-op; P04 is now real-data only")
    parser.add_argument("--smoke", action="store_true", help="Fast GRD-only smoke run with a tiny model")
    parser.add_argument("--data_dir", type=str, default=None, help="HDF5 data directory (default: ./data)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint/artifact directory (default: ./artifacts)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint to load")
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=1000,
        help="Number of test samples to evaluate; 0 or negative means full test set",
    )
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA automatic mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_filters", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=17)
    parser.add_argument("--w_char", type=float, default=0.8)
    parser.add_argument("--w_ssim", type=float, default=0.2)

    # Data-generation options passed through to generate_data.py.
    parser.add_argument("--data_root", type=str, default=os.environ.get(
        "P04_SAR_DATA_ROOT",
        os.path.join(os.path.dirname(__file__), "raw_sentinel1"),
    ))
    parser.add_argument("--grd_path", type=str, default=None)
    parser.add_argument("--slc_dir", action="append", default=None)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--look_size", type=int, default=4)
    parser.add_argument("--smooth_method", choices=["multilook", "gaussian"], default="multilook")
    return parser


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args = _build_parser().parse_args()
    root = Path(__file__).resolve().parent
    args.data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    args.ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else root / "artifacts"
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.epochs = 2
        args.batch_size = 4
        if args.eval_samples > 0:
            args.eval_samples = min(args.eval_samples, 20)
        args.num_workers = 0
        args.n_filters = 16
        args.n_layers = 5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.generate or args.generate_real:
        _generate_real_data(args)

    train_path = args.data_dir / "real_despeckling_train.h5"
    val_path = args.data_dir / "real_despeckling_val.h5"
    test_path = args.data_dir / "real_despeckling_test.h5"
    missing = [p for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        print("ERROR: missing real Sentinel-1 HDF5 files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print("Run with --generate, or point --data_dir at a prebuilt real_despeckling_* dataset.", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    print("=== P04: DnCNN-SAR Despeckling (Real Sentinel-1) ===")
    print(f"  Device:     {device}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Ckpt dir:   {args.ckpt_dir}")
    print(f"  Model:      filters={args.n_filters}, layers={args.n_layers}")
    print(f"  Epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"  AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"  Loss: Charbonnier={args.w_char}, SSIM={args.w_ssim}")
    print()

    model = DnCNNSAR(n_channels=1, n_filters=args.n_filters, n_layers=args.n_layers).to(device)
    print(f"  DnCNN-SAR parameters: {count_parameters(model):,}")

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
            return 1
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Loaded checkpoint: {ckpt}")

    train_ds = RealDespecklingDataset(train_path, augment=not args.eval_only)
    val_ds = RealDespecklingDataset(val_path, augment=False)
    test_ds = RealDespecklingDataset(test_path, augment=False)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    if args.eval_only:
        print("\n=== Evaluation Only ===")
        summary = evaluate_full(
            model,
            test_ds,
            device,
            eval_samples=args.eval_samples,
            checkpoint_path=args.checkpoint,
            data_path=test_path,
            output_csv=args.ckpt_dir / "per_sample_metrics.csv",
            config=vars(args),
        )
        _print_summary(summary)
        _save_summary(summary, args.ckpt_dir / "eval_results.json")
        return 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    criterion = DespecklingLoss(w_char=args.w_char, w_ssim=args.w_ssim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    best_val_psnr = -float("inf")
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "lr": []}

    print(f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>11}  {'Val PSNR':>9}  {'LR':>10}  {'Time':>6}")
    print("-" * 62)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss, val_psnr = validate_psnr(model, val_loader, criterion, device, use_amp)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["lr"].append(lr)

        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), args.ckpt_dir / "best_model.pt")

        elapsed = time.time() - t0
        marker = " *" if is_best else ""
        print(
            f"{epoch:>5d}  {train_loss:>11.6f}  {val_loss:>11.6f}  "
            f"{val_psnr:>9.3f}  {lr:>10.2e}  {elapsed:>5.1f}s{marker}"
        )

        if epoch % 20 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.ckpt_dir / f"checkpoint_ep{epoch:03d}.pt")

    with (args.ckpt_dir / "history.json").open("w") as f:
        json.dump(history, f)
    print(f"\nBest val PSNR: {best_val_psnr:.3f} dB")

    print("\n=== Test Set Evaluation ===")
    best_ckpt = args.ckpt_dir / "best_model.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    summary = evaluate_full(
        model,
        test_ds,
        device,
        eval_samples=args.eval_samples,
        checkpoint_path=best_ckpt,
        data_path=test_path,
        output_csv=args.ckpt_dir / "per_sample_metrics.csv",
        config=vars(args),
    )
    _print_summary(summary)
    _save_summary(summary, args.ckpt_dir / "eval_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
