#!/usr/bin/env python3
"""Save example P05 STFT images for quick inspection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.waveform_library import WAVEFORM_CLASSES, WaveformConfig, generate_waveform_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts/examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    config = WaveformConfig()
    for class_name in WAVEFORM_CLASSES:
        image, _, _, meta = generate_waveform_sample(class_name, rng, config, snr_db=6.0)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"{class_name}  SNR={meta['snr_db']:.1f} dB")
        ax.set_xlabel("time frame")
        ax.set_ylabel("frequency bin")
        fig.tight_layout()
        path = out_dir / f"p05_{class_name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
