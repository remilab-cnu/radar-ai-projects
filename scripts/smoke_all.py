#!/usr/bin/env python3
"""Run smoke tests for all projects to verify they work end-to-end."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = REPO_ROOT / "projects"


def main():
    projects = sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir() and p.name.startswith("p"))
    results = {}

    for proj in projects:
        train_py = proj / "train.py"
        if not train_py.exists():
            results[proj.name] = "SKIP (no train.py)"
            continue

        print(f"\n{'='*60}")
        print(f"  Smoke test: {proj.name}")
        print(f"{'='*60}")

        ret = subprocess.run(
            [sys.executable, str(train_py), "--generate", "--smoke"],
            cwd=str(proj),
            timeout=300,
        )
        results[proj.name] = "PASS" if ret.returncode == 0 else f"FAIL (exit {ret.returncode})"

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
