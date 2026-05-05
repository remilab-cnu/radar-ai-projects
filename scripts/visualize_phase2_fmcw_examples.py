#!/usr/bin/env python3
"""Create deterministic shared FMCW simulator example plots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.fmcw_simulator import (  # noqa: E402
    FMCWRadar,
    generate_scene,
    range_axis,
    range_doppler_map,
    range_fft,
    to_db,
    velocity_axis,
)
from projects.p03_radar_cube_doa import generate_data as p03  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _range_fft_plot(out_dir: Path) -> Path:
    radar = FMCWRadar(fc=9.6e9, bw=20e6, T_chirp=6.4e-6, PRI=100e-6, N_chirps=64, N_samples=512)
    targets = [
        {"range": 750.0, "velocity": 0.0, "rcs": 1.0},
        {"range": 1125.0, "velocity": 0.0, "rcs": 0.35},
    ]
    raw, meta = generate_scene(radar, targets, snr_db=28.0, seed=20260429, return_meta=True)
    rc = range_fft(raw[0], radar=radar, window="hann")
    profile = np.mean(np.abs(rc), axis=0)
    x = range_axis(radar)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, to_db(profile), lw=1.4)
    for info in meta["target_info"]:
        ax.axvline(info["range_bin"] * radar.range_bin_spacing, color="tab:red", ls="--", alpha=0.75)
        ax.text(info["range_bin"] * radar.range_bin_spacing, -6, f"{info['range']:.0f} m", rotation=90,
                va="top", ha="right", fontsize=8, color="tab:red")
    ax.set_xlim(500, 1300)
    ax.set_ylim(-70, 3)
    ax.set_title("FMCW dechirp 1D range FFT: beat peaks from delayed chirps")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Magnitude [dB, peak-ref]")
    ax.grid(True, alpha=0.25)
    path = out_dir / "01_range_fft.png"
    _save(fig, path)
    return path


def _p01_rdm_plot(out_dir: Path) -> Path:
    radar = FMCWRadar(fc=9.6e9, bw=20e6, T_chirp=6.4e-6, PRI=100e-6, N_chirps=128, N_samples=512)
    targets = [
        {"range": 420.0, "velocity": -18.0, "rcs": 250.0},
        {"range": 870.0, "velocity": 12.0, "rcs": 180.0},
        {"range": 1280.0, "velocity": -34.0, "rcs": 400.0},
        {"range": 1640.0, "velocity": 28.0, "rcs": 320.0},
    ]
    signal, meta = generate_scene(radar, targets, snr_db=24.0, seed=1001, return_meta=True)
    target_info = meta["target_info"]
    rdm = range_doppler_map(signal[0:1], radar=radar, window_range="hann", window_doppler="hann")
    mag = np.abs(rdm[0, :, :radar.N_range_bins])
    db = to_db(mag)
    x = range_axis(radar)
    v = velocity_axis(radar)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    im = ax.imshow(db, origin="lower", aspect="auto", extent=[x[0], x[-1], v[0], v[-1]],
                   cmap="viridis", vmin=-55, vmax=0)
    for info in target_info:
        ax.plot(info["range_bin"] * radar.range_bin_spacing, v[info["doppler_bin"]], "rx", ms=7, mew=1.5)
    ax.set_title("P1-style FMCW dechirped RDM with visible target labels")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Velocity [m/s]")
    fig.colorbar(im, ax=ax, label="Magnitude [dB, peak-ref]")
    path = out_dir / "02_p01_fmcw_rdm.png"
    _save(fig, path)
    return path


def _p03_rdm_plot(out_dir: Path) -> Path:
    rng = np.random.default_rng(333)
    rd_cube, meta = p03._simulate_raw_cube(rng)  # noqa: SLF001 - visualization probe
    mag = np.sum(np.abs(rd_cube), axis=0)
    db = to_db(mag)
    x = p03.RANGE_AXIS_M
    v = p03.VELOCITY_AXIS_MPS

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    im = ax.imshow(db, origin="lower", aspect="auto", extent=[x[0], x[-1], v[0], v[-1]],
                   cmap="magma", vmin=-55, vmax=0)
    ax.plot(meta["range_m"], meta["velocity_mps"], "cx", ms=8, mew=1.8)
    ax.set_title(f"P3 shared-FMCW RD map before antenna-vector selection, angle={meta['angle_deg']:.1f}°")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Velocity [m/s]")
    fig.colorbar(im, ax=ax, label="Magnitude [dB, peak-ref]")
    path = out_dir / "03_p03_selected_target_rdm.png"
    _save(fig, path)
    return path


def _range_walk_diagnostic(out_dir: Path) -> Path:
    radar = FMCWRadar(fc=9.6e9, bw=20e6, T_chirp=6.4e-6, PRI=200e-6, N_chirps=128, N_samples=512)
    targets = [
        {"range": 900.0, "velocity": -35.0, "rcs": 1.0},
        {"range": 900.0, "velocity": 0.0, "rcs": 1.0},
        {"range": 900.0, "velocity": 35.0, "rcs": 1.0},
    ]
    raw, _ = generate_scene(radar, targets, snr_db=35.0, seed=4242, return_meta=True)
    rdm = range_doppler_map(raw, radar=radar, window_range="hann", window_doppler="hann")[0]
    db = to_db(np.abs(rdm))
    x = range_axis(radar)
    v = velocity_axis(radar)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    im = ax.imshow(db, origin="lower", aspect="auto", extent=[x[0], x[-1], v[0], v[-1]],
                   cmap="viridis", vmin=-55, vmax=0)
    ax.axvline(900.0, color="white", ls="--", lw=1.1, alpha=0.8, label="initial range")
    ax.set_xlim(820, 980)
    ax.set_ylim(-45, 45)
    ax.set_title("Range-walk/Doppler diagnostic from moving FMCW echoes")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Velocity [m/s]")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="Magnitude [dB, peak-ref]")
    path = out_dir / "05_range_walk_diagnostic.png"
    _save(fig, path)
    return path


def _write_html(out_dir: Path, images: list[Path]) -> Path:
    captions = {
        "01_range_fft.png": "1D range FFT from explicit FMCW dechirp/mixing.",
        "02_p01_fmcw_rdm.png": "P1-style range-Doppler map after FMCW range FFT and Doppler FFT.",
        "03_p03_selected_target_rdm.png": "P3 shared-FMCW RD map before simulator-known antenna-vector selection.",
        "05_range_walk_diagnostic.png": "Moving-target diagnostic: range walk and Doppler sidelobes arise from the echo model and processing chain.",
    }
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Shared FMCW simulator examples</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.5} img{max-width:100%;border:1px solid #ddd;border-radius:8px} figure{margin:28px 0} code{background:#f5f5f5;padding:2px 5px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>Shared FMCW dechirp simulator examples</h1>",
        "<p>Deterministic examples generated from <code>shared/fmcw_simulator.py</code> from explicit complex-baseband transmit/receive chirps, mixer/dechirp, range FFT, and Doppler FFT. RF up/down conversion is excluded.</p>",
    ]
    for image in images:
        html += ["<figure>", f"<img src='{image.name}' alt='{image.stem}'>", f"<figcaption>{captions.get(image.name, image.name)}</figcaption>", "</figure>"]
    html += ["</body></html>"]
    path = out_dir / "index.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=ROOT / "docs" / "technical" / "fmcw_dechirp_examples")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = [
        _range_fft_plot(args.out_dir),
        _p01_rdm_plot(args.out_dir),
        _p03_rdm_plot(args.out_dir),
        _range_walk_diagnostic(args.out_dir),
    ]
    html = _write_html(args.out_dir, images)
    print("Wrote:")
    for path in images + [html]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
