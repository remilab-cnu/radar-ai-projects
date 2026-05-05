#!/usr/bin/env python3
"""Pick a few representative test scenes and dump everything the figure
script needs for same-scene comparison: RDM, ground truth mask, CFAR
detections, U-Net probability map (and binary detections at locked
threshold). Writes a single .npz per scene.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import torch

from eval_utils import assert_schema_v2, load_policy, split_path
from model import UNetDetector
from shared.fmcw_simulator import ca_cfar_2d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cfar_policy", required=True)
    ap.add_argument("--unet_policy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_scenes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfar_pol = load_policy(args.cfar_policy)
    unet_pol = load_policy(args.unet_policy)
    guard = tuple(cfar_pol["guard"])
    train = tuple(cfar_pol["train"])
    pfa = float(cfar_pol.get("pfa_design", cfar_pol.get("pfa", 1e-4)))
    threshold = float(unet_pol["threshold"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetDetector(in_channels=2, base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    path = split_path(args.data_dir, args.split)
    summary = []
    with h5py.File(path, "r") as f:
        assert_schema_v2(f)
        n = len(f["x"])
        # Pick scenes spanning SNR range — sort by SNR and sample uniformly
        snr_all = f["snr_db"][:]
        order = np.argsort(snr_all)
        picks = [int(order[int(round(p))]) for p in
                 np.linspace(0, n - 1, args.n_scenes)]

        range_axis = f["range_axis_m"][:]
        velocity_axis = f["velocity_axis_mps"][:]

        for k, idx in enumerate(picks):
            x = f["x"][idx]
            gt = f["y"][idx, 0] > 0.5
            rdm_mag = f["rdm_mag_linear"][idx]
            snr = float(f["snr_db"][idx])
            n_targets = int(f["n_targets"][idx])
            target_range = f["target_range_m"][idx]
            target_velocity = f["target_velocity_mps"][idx]
            target_rcs = f["target_rcs"][idx]

            cfar_det = ca_cfar_2d(rdm_mag, guard=guard, train=train, pfa=pfa)
            with torch.no_grad():
                tx = torch.as_tensor(x[None], dtype=torch.float32, device=device)
                prob = model(tx).cpu().numpy()[0, 0]
            unet_det = prob > threshold

            np.savez(
                out_dir / f"scene_{k:02d}.npz",
                rdm_log_mag=x[0],          # noise-floor referenced [0,1]
                rdm_mag_linear=rdm_mag,
                gt_mask=gt.astype(np.uint8),
                cfar_det=cfar_det.astype(np.uint8),
                unet_prob=prob,
                unet_det=unet_det.astype(np.uint8),
                snr_db=snr,
                n_targets=n_targets,
                target_range=target_range,
                target_velocity=target_velocity,
                target_rcs=target_rcs,
                range_axis_m=range_axis,
                velocity_axis_mps=velocity_axis,
                cfar_guard=np.array(guard),
                cfar_train=np.array(train),
                cfar_pfa=np.array([pfa]),
                unet_threshold=np.array([threshold]),
            )
            summary.append({
                "scene": k,
                "test_idx": idx,
                "snr_db": snr,
                "n_targets": n_targets,
                "cfar_n_detections": int(cfar_det.sum()),
                "unet_n_detections": int(unet_det.sum()),
            })

    (out_dir / "scenes_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved {args.n_scenes} scenes to {out_dir}")


if __name__ == "__main__":
    main()
