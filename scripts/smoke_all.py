#!/usr/bin/env python3
"""Run smoke tests for active projects to verify they work end-to-end.

Usage:
    python scripts/smoke_all.py           # --generate --smoke (fast, default)
    python scripts/smoke_all.py --full    # --generate --epochs 2 (more thorough)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = REPO_ROOT / "projects"

ACTIVE_PROJECTS = [
    "p01_unet_detector",
    "p02_resnet18_har",
    "p03_radar_cube_doa",
    "p04_dncnn_sar",
]


def main():
    parser = argparse.ArgumentParser(description="Smoke-test all radar-AI projects.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run --generate --epochs 2 instead of --generate --smoke for a more thorough test.",
    )
    args = parser.parse_args()

    # Assert all active project directories exist before running.
    missing = [p for p in ACTIVE_PROJECTS if not (PROJECTS_DIR / p).is_dir()]
    if missing:
        print(f"ERROR: Missing project directories: {missing}", file=sys.stderr)
        sys.exit(1)

    default_train_args = ["--generate", "--epochs", "2"] if args.full else ["--generate", "--smoke"]

    results = {}
    for name in ACTIVE_PROJECTS:
        proj = PROJECTS_DIR / name
        train_py = proj / "train.py"
        if not train_py.exists():
            results[name] = "SKIP (no train.py)"
            continue

        print(f"\n{'='*60}")
        print(f"  Smoke test: {name}")
        print(f"{'='*60}")

        train_args = list(default_train_args)
        if name == "p03_radar_cube_doa":
            train_args = (
                ["--mapping", "--generate", "--epochs", "2"]
                if args.full else
                ["--mapping", "--generate", "--smoke"]
            )
        if name == "p04_dncnn_sar" and args.full:
            # P04 is the real-Sentinel-1 lecture project. A non-smoke
            # --generate run scans focused SLC products and is a full
            # experiment setup, not a 300-second all-project smoke check.
            train_args = ["--generate", "--smoke"]
            print("  Note: using --smoke for P04 even under --full; full SLC experiments require instructor-provided data and a longer run.")

        ret = subprocess.run(
            [sys.executable, str(train_py)] + train_args,
            cwd=str(proj),
            timeout=300,
        )
        results[name] = "PASS" if ret.returncode == 0 else f"FAIL (exit {ret.returncode})"

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "v" if status == "PASS" else ("~" if "SKIP" in status else "x")
        print(f"  [{icon}] {name}: {status}")

    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
