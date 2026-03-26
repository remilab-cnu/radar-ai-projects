"""HDF5 dataset I/O helpers."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def save_hdf5(path: str | Path, **arrays: np.ndarray) -> None:
    """Save arrays to an HDF5 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in arrays.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
    print(f"  Saved {path.name}: {', '.join(f'{k} {v.shape}' for k, v in arrays.items())}")


def load_hdf5(path: str | Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load arrays from an HDF5 file."""
    with h5py.File(path, "r") as f:
        if keys is None:
            keys = list(f.keys())
        return {k: f[k][:] for k in keys}


class HDF5Dataset(Dataset):
    """Generic HDF5 dataset. Returns (x_tensor, y_tensor)."""

    def __init__(self, path: str | Path, x_key: str = "x", y_key: str = "y",
                 x_dtype=torch.float32, y_dtype=torch.float32):
        data = load_hdf5(path, [x_key, y_key])
        self.x = torch.as_tensor(data[x_key], dtype=x_dtype)
        self.y = torch.as_tensor(data[y_key], dtype=y_dtype)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
