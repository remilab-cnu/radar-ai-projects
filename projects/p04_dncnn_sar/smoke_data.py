from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Final

import h5py
import numpy as np
from numpy.typing import NDArray


FloatImage = NDArray[np.float32]

SOURCE_LABEL: Final = b"synthetic_smoke"
DATA_TYPE: Final = "synthetic_p04_smoke"
DYNAMIC_RANGE_DB: Final = 35.0
INTENSITY_FLOOR: Final = 1.0e-6
MIN_PATCH_SIZE: Final = 16
BUNDLED_REAL_SMOKE_DIR: Final = Path(__file__).with_name("sample_data")
SPLIT_FILENAMES: Final = (
    "real_despeckling_train.h5",
    "real_despeckling_val.h5",
    "real_despeckling_test.h5",
)


@dataclass(frozen=True, slots=True)
class SyntheticSplitCounts:
    train: int = 24
    val: int = 6
    test: int = 6


@dataclass(frozen=True, slots=True)
class InvalidSyntheticPatchSizeError(ValueError):
    patch_size: int

    def __str__(self) -> str:
        return (
            f"synthetic P04 smoke patch_size must be at least {MIN_PATCH_SIZE}, "
            f"got {self.patch_size}"
        )


DEFAULT_SPLIT_COUNTS: Final = SyntheticSplitCounts()


def bundled_real_smoke_available() -> bool:
    return all((BUNDLED_REAL_SMOKE_DIR / filename).exists() for filename in SPLIT_FILENAMES)


def copy_bundled_real_smoke_splits(out_dir: str | Path) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in SPLIT_FILENAMES:
        src = BUNDLED_REAL_SMOKE_DIR / filename
        dst = output_dir / filename
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024**2)
        print(f"  Copied {dst} ({size_mb:.1f} MB)")


def _coordinate_grid(patch_size: int) -> tuple[FloatImage, FloatImage]:
    axis = np.linspace(-1.0, 1.0, patch_size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    return grid_y.astype(np.float32), grid_x.astype(np.float32)


def _reflectivity_scene(
    rng: np.random.Generator,
    grid_y: FloatImage,
    grid_x: FloatImage,
) -> FloatImage:
    scene = np.full(grid_x.shape, 0.06, dtype=np.float32)
    scene += (0.03 * (grid_x + 1.0)).astype(np.float32)
    scene += (0.02 * (grid_y + 1.0)).astype(np.float32)

    for _ in range(4):
        center_x = float(rng.uniform(-0.75, 0.75))
        center_y = float(rng.uniform(-0.75, 0.75))
        width_x = float(rng.uniform(0.05, 0.20))
        width_y = float(rng.uniform(0.05, 0.20))
        amplitude = float(rng.uniform(0.25, 1.0))
        blob = np.exp(
            -(
                ((grid_x - center_x) ** 2) / (2.0 * width_x**2)
                + ((grid_y - center_y) ** 2) / (2.0 * width_y**2)
            )
        )
        scene += (amplitude * blob).astype(np.float32)

    slope = float(rng.uniform(-0.8, 0.8))
    offset = float(rng.uniform(-0.35, 0.35))
    bright_side = grid_y > (slope * grid_x + offset)
    scene += (0.12 * bright_side.astype(np.float32)).astype(np.float32)
    return np.clip(scene, INTENSITY_FLOOR, None).astype(np.float32)


def _to_log_unit(intensity: FloatImage) -> FloatImage:
    peak = max(float(np.max(intensity)), INTENSITY_FLOOR)
    safe = np.maximum(intensity, INTENSITY_FLOOR)
    db = 10.0 * np.log10(safe / peak)
    clipped = np.clip(db, -DYNAMIC_RANGE_DB, 0.0)
    return ((clipped + DYNAMIC_RANGE_DB) / DYNAMIC_RANGE_DB).astype(np.float32)


def _make_patch(
    rng: np.random.Generator,
    grid_y: FloatImage,
    grid_x: FloatImage,
) -> tuple[FloatImage, FloatImage]:
    clean_intensity = _reflectivity_scene(rng, grid_y, grid_x)
    speckle = rng.gamma(shape=1.0, scale=1.0, size=clean_intensity.shape)
    noisy_intensity = clean_intensity * speckle.astype(np.float32)
    noisy = _to_log_unit(noisy_intensity.astype(np.float32))
    clean = _to_log_unit(clean_intensity)
    return noisy, clean


def _write_split(path: Path, sample_count: int, patch_size: int, rng: np.random.Generator) -> None:
    grid_y, grid_x = _coordinate_grid(patch_size)
    chunk = min(8, sample_count)

    with h5py.File(path, "w") as handle:
        noisy_ds = handle.create_dataset(
            "noisy",
            shape=(sample_count, 1, patch_size, patch_size),
            dtype="float32",
            chunks=(chunk, 1, patch_size, patch_size),
            compression="gzip",
            compression_opts=4,
        )
        clean_ds = handle.create_dataset(
            "clean",
            shape=(sample_count, 1, patch_size, patch_size),
            dtype="float32",
            chunks=(chunk, 1, patch_size, patch_size),
            compression="gzip",
            compression_opts=4,
        )
        source_ds = handle.create_dataset(
            "source",
            shape=(sample_count,),
            dtype=h5py.special_dtype(vlen=bytes),
        )

        for idx in range(sample_count):
            noisy, clean = _make_patch(rng, grid_y, grid_x)
            noisy_ds[idx] = noisy[np.newaxis, :, :]
            clean_ds[idx] = clean[np.newaxis, :, :]
            source_ds[idx] = SOURCE_LABEL

        handle.attrs["n_samples"] = sample_count
        handle.attrs["patch_size"] = patch_size
        handle.attrs["data_type"] = DATA_TYPE
        handle.attrs["source_note"] = (
            "Synthetic clone-safe smoke data; do not use for P04 Sentinel-1 result claims."
        )


def write_synthetic_despeckling_splits(
    out_dir: str | Path,
    *,
    patch_size: int = 256,
    seed: int = 42,
    counts: SyntheticSplitCounts = DEFAULT_SPLIT_COUNTS,
) -> None:
    if patch_size < MIN_PATCH_SIZE:
        raise InvalidSyntheticPatchSizeError(patch_size)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    split_specs = (
        ("train", counts.train),
        ("val", counts.val),
        ("test", counts.test),
    )
    for split_name, sample_count in split_specs:
        path = output_dir / f"real_despeckling_{split_name}.h5"
        _write_split(path, sample_count, patch_size, rng)
        size_mb = path.stat().st_size / (1024**2)
        print(f"  Saved {path} ({sample_count} synthetic smoke samples, {size_mb:.1f} MB)")
