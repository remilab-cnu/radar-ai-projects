#!/usr/bin/env python3
"""Save example P06 target signatures for quick inspection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.target_signature import TARGET_CLASSES, TargetSignatureConfig, generate_target_signature_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts/examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    config = TargetSignatureConfig()
    for class_name in TARGET_CLASSES:
        tensor, _, _, meta = generate_target_signature_sample(class_name, rng, config, snr_db=12.0)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(tensor[0], label="magnitude")
        ax.plot(tensor[1], label="phase", alpha=0.75)
        ax.set_title(f"{class_name}  aspect={meta['center_aspect_deg']:.1f} deg  SNR={meta['snr_db']:.1f} dB")
        ax.set_xlabel("sample")
        ax.set_ylabel("normalized value")
        ax.legend(loc="best")
        fig.tight_layout()
        path = out_dir / f"p06_{class_name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
