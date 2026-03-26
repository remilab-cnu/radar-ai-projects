"""Standard CLI argument parser shared across all projects."""
from __future__ import annotations

import argparse


def base_parser(description: str = "") -> argparse.ArgumentParser:
    """Return a parser with the standard flags every project supports."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--generate", action="store_true", help="Generate HDF5 datasets before training")
    p.add_argument("--smoke", action="store_true", help="Quick smoke test (tiny data, 2 epochs)")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--eval_only", action="store_true", help="Skip training, run evaluation only")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p
