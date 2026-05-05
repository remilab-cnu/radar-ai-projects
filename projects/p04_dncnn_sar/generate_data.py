#!/usr/bin/env python3
"""Generate P04 despeckling datasets from real Sentinel-1 SAR data.

Two data sources:
1. GRD: db (noisy) + db_filtered (clean) -> direct pair
2. SLC: complex image -> |mag| log-dB = noisy, spatial multi-look = pseudo-clean

Both are cut into 256x256 patches, normalized to [0,1], and stored in HDF5.

Usage:
    python generate_data.py
    python generate_data.py --out_dir /custom/path
    python generate_data.py --smoke   # quick GRD-only test

Output:
    {out_dir}/real_despeckling_train.h5
    {out_dir}/real_despeckling_val.h5
    {out_dir}/real_despeckling_test.h5

HDF5 Schema (per split):
    'noisy'  : (N, 1, 256, 256) float32 — speckled SAR [0, 1]
    'clean'  : (N, 1, 256, 256) float32 — pseudo-clean SAR [0, 1]
    'source' : (N,) bytes          — b'grd' or b'slc'
"""

import os
import argparse

import h5py
import numpy as np

# ─── Data Paths (override with P04_SAR_DATA_ROOT or --data_root) ────────────

DEFAULT_DATA_ROOT = os.environ.get(
    "P04_SAR_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "raw_sentinel1"),
)

GRD_REL_PATH = (
    "sentinel1_grd_baseline/processed/"
    "S1A_IW_GRDH_1SDV_20250127T211653_20250127T211722_057635_071A24_vv_crop2048.npz"
)

SLC_REL_DIRS = [
    "sentinel1_l0/official_focus_stageA_smoketest_cd6a",
    "sentinel1_l0/official_focus_sw10_tops_dem",
    "sentinel1_l0/official_focus_sw10",
    # Batch-focused acquisitions (2026-03-15)
    "sentinel1_l0/focused_vv_41d7_allchunks",
    "sentinel1_l0/focused_vv_9081_allchunks",
    "sentinel1_l0/focused_vv_bc54_allchunks",
]


def _resolve_default_paths(data_root: str):
    """Return the canonical GRD path and SLC directories for a data root."""
    grd_path = os.path.join(data_root, GRD_REL_PATH)
    slc_dirs = [os.path.join(data_root, rel) for rel in SLC_REL_DIRS]
    return grd_path, slc_dirs


GRD_PATH, SLC_DIRS = _resolve_default_paths(DEFAULT_DATA_ROOT)


# ─── GRD Patch Extraction ────────────────────────────────────────────────────

def extract_grd_patches(patch_size=256, overlap_stride=128):
    """Extract noisy/clean patch pairs from GRD data.

    Non-overlapping patches (stride=patch_size) + overlapping patches
    (stride=overlap_stride) are both generated.

    Returns list of (noisy_patch, clean_patch) tuples, all float32 [0, 1].
    """
    if not os.path.exists(GRD_PATH):
        raise FileNotFoundError(
            f"GRD source not found: {GRD_PATH}\n"
            "Set --data_root/--grd_path or P04_SAR_DATA_ROOT to the Sentinel-1 data root."
        )

    print(f"  Loading GRD: {GRD_PATH}")
    data = np.load(GRD_PATH)
    db = data["db"].astype(np.float64)              # (2048, 2048)
    db_filtered = data["db_filtered"].astype(np.float64)  # (2048, 2048)

    # Normalize both to [0, 1] using db's range as reference
    vmin, vmax = db.min(), db.max()
    print(f"  GRD db range: [{vmin:.2f}, {vmax:.2f}] dB")

    if vmax - vmin < 1e-6:
        print("  WARNING: GRD db range too small, skipping")
        return []

    db_norm = (db - vmin) / (vmax - vmin)
    db_filt_norm = (db_filtered - vmin) / (vmax - vmin)
    db_filt_norm = np.clip(db_filt_norm, 0.0, 1.0)

    H, W = db_norm.shape
    patches = []

    # Non-overlapping patches (stride = patch_size)
    n_nonoverlap = 0
    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            noisy_patch = db_norm[r:r + patch_size, c:c + patch_size]
            clean_patch = db_filt_norm[r:r + patch_size, c:c + patch_size]
            patches.append((noisy_patch.astype(np.float32),
                            clean_patch.astype(np.float32)))
            n_nonoverlap += 1
    print(f"  GRD non-overlapping patches: {n_nonoverlap}")

    # Overlapping patches (only if stride < patch_size)
    if overlap_stride < patch_size:
        # Collect non-overlap positions to skip duplicates
        nonoverlap_positions = set()
        for r in range(0, H - patch_size + 1, patch_size):
            for c in range(0, W - patch_size + 1, patch_size):
                nonoverlap_positions.add((r, c))

        n_overlap = 0
        for r in range(0, H - patch_size + 1, overlap_stride):
            for c in range(0, W - patch_size + 1, overlap_stride):
                if (r, c) in nonoverlap_positions:
                    continue  # already included
                noisy_patch = db_norm[r:r + patch_size, c:c + patch_size]
                clean_patch = db_filt_norm[r:r + patch_size, c:c + patch_size]
                patches.append((noisy_patch.astype(np.float32),
                                clean_patch.astype(np.float32)))
                n_overlap += 1
        print(f"  GRD overlapping patches (stride {overlap_stride}): {n_overlap} additional")

    print(f"  GRD total patches: {len(patches)}")
    return patches


# ─── SLC Patch Extraction ────────────────────────────────────────────────────

def spatial_multilook(mag, look_size=4):
    """Apply spatial multi-look averaging on magnitude image.

    Averages non-overlapping look_size x look_size blocks, then upsamples
    back to original resolution via nearest-neighbor to keep patch size.

    Parameters
    ----------
    mag : ndarray (H, W) — magnitude image (linear scale)
    look_size : int — averaging block size

    Returns
    -------
    smoothed : ndarray (H, W) — multi-looked magnitude, same shape as input
    """
    H, W = mag.shape
    # Trim to multiple of look_size
    Ht = (H // look_size) * look_size
    Wt = (W // look_size) * look_size
    trimmed = mag[:Ht, :Wt]

    # Reshape and average
    blocks = trimmed.reshape(Ht // look_size, look_size,
                             Wt // look_size, look_size)
    averaged = blocks.mean(axis=(1, 3))  # (Ht/L, Wt/L)

    # Upsample back to original size via repeat
    smoothed = np.repeat(np.repeat(averaged, look_size, axis=0),
                         look_size, axis=1)

    # Pad if trimmed
    if smoothed.shape[0] < H or smoothed.shape[1] < W:
        result = np.zeros((H, W), dtype=smoothed.dtype)
        result[:smoothed.shape[0], :smoothed.shape[1]] = smoothed
        # Fill edges by repeating last row/col
        if smoothed.shape[0] < H:
            result[smoothed.shape[0]:, :smoothed.shape[1]] = smoothed[-1:, :]
        if smoothed.shape[1] < W:
            result[:, smoothed.shape[1]:] = result[:, smoothed.shape[1] - 1:smoothed.shape[1]]
        return result
    return smoothed


def slc_to_noisy_clean(img_complex, look_size=4, clip_db=40.0, smooth_method='multilook'):
    """Convert complex SLC image to noisy/clean patch pair in normalized dB.

    noisy = 20*log10(|img| / max + eps), clipped to [-clip_db, 0], normalized to [0, 1]
    clean = same transform applied to smoothed magnitude

    Parameters
    ----------
    img_complex : ndarray (H, W) complex64
    look_size : int — block size for multi-look (only used if smooth_method='multilook')
    clip_db : float — dynamic range floor in dB
    smooth_method : str — 'multilook' (lecture default) or 'gaussian'
        'multilook': block-average multi-look on intensity (physical P04 v8 target)
        'gaussian': Gaussian smoothing on intensity (v9 ablation; easier but blurrier target)

    Returns
    -------
    noisy_norm, clean_norm : ndarray (H, W) float32, both in [0, 1]
    """
    mag = np.abs(img_complex).astype(np.float64)
    mag_max = mag.max()
    if mag_max < 1e-15:
        return None, None  # empty image

    # Noisy: original magnitude in log-dB
    noisy_db = 20.0 * np.log10(mag / mag_max + 1e-10)
    noisy_db = np.clip(noisy_db, -clip_db, 0.0)
    noisy_norm = (noisy_db + clip_db) / clip_db  # [-clip_db, 0] -> [0, 1]

    # Clean: smoothed in INTENSITY domain (physically correct for speckle)
    # Speckle is multiplicative on intensity I=|S|^2, so averaging must be on intensity.
    intensity = mag ** 2
    intensity_max = mag_max ** 2
    if smooth_method == 'gaussian':
        from scipy.ndimage import gaussian_filter
        sigma = look_size / 3.5  # sigma≈1.14 for look_size=4, matches ~16 effective looks
        intensity_smooth = gaussian_filter(intensity, sigma=sigma)
    else:
        intensity_smooth = spatial_multilook(intensity, look_size=look_size)
    # Convert back: 10*log10(I/I_max) = 20*log10(sqrt(I)/mag_max)
    clean_db = 10.0 * np.log10(intensity_smooth / intensity_max + 1e-10)
    clean_db = np.clip(clean_db, -clip_db, 0.0)
    clean_norm = (clean_db + clip_db) / clip_db

    return noisy_norm.astype(np.float32), clean_norm.astype(np.float32)


def extract_slc_patches_from_file(npz_path, patch_size=256, look_size=4, smooth_method='multilook'):
    """Extract noisy/clean patches from a single SLC .npz file.

    Returns list of (noisy_patch, clean_patch) tuples.
    """
    data = np.load(npz_path)
    img = data["img"]  # complex64

    noisy_full, clean_full = slc_to_noisy_clean(img, look_size=look_size, smooth_method=smooth_method)
    if noisy_full is None:
        return []

    H, W = noisy_full.shape
    patches = []

    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            noisy_p = noisy_full[r:r + patch_size, c:c + patch_size]
            clean_p = clean_full[r:r + patch_size, c:c + patch_size]

            # Skip nearly-empty patches (too dark = no signal)
            if noisy_p.mean() < 0.05:
                continue

            patches.append((noisy_p, clean_p))

    return patches


def extract_all_slc_patches(patch_size=256, look_size=4, smooth_method='multilook'):
    """Extract patches from all available SLC data files.

    Returns list of (noisy_patch, clean_patch) tuples.
    """
    all_patches = []

    for slc_dir in SLC_DIRS:
        if not os.path.isdir(slc_dir):
            print(f"  SLC dir not found, skipping: {slc_dir}")
            continue

        dir_name = os.path.basename(slc_dir)
        dir_patches = 0

        # Find all focused .npz files
        for fname in sorted(os.listdir(slc_dir)):
            if not fname.endswith("_focused.npz"):
                continue

            fpath = os.path.join(slc_dir, fname)
            patches = extract_slc_patches_from_file(
                fpath, patch_size=patch_size, look_size=look_size,
                smooth_method=smooth_method)
            all_patches.extend(patches)
            dir_patches += len(patches)
            print(f"    {dir_name}/{fname}: {len(patches)} patches")

        print(f"  {dir_name} total: {dir_patches} patches")

    print(f"  SLC total patches: {len(all_patches)}")
    return all_patches


# ─── Quality Filter ──────────────────────────────────────────────────────────

def filter_patches(patches, min_std=0.02, min_mean=0.05, max_mean=0.98):
    """Remove low-quality patches (too flat, too dark, or saturated).

    Parameters
    ----------
    patches : list of (noisy, clean) tuples
    min_std : float — minimum std of noisy patch
    min_mean : float — minimum mean of noisy patch
    max_mean : float — maximum mean of noisy patch

    Returns
    -------
    filtered : list of (noisy, clean) tuples
    """
    filtered = []
    for noisy, clean in patches:
        std = noisy.std()
        mean = noisy.mean()
        if std >= min_std and min_mean <= mean <= max_mean:
            filtered.append((noisy, clean))

    n_removed = len(patches) - len(filtered)
    if n_removed > 0:
        print(f"  Filtered out {n_removed} low-quality patches "
              f"({len(filtered)} remaining)")
    return filtered


# ─── Save to HDF5 ────────────────────────────────────────────────────────────

def save_split(path, noisy_list, clean_list, source_list, *, smooth_method, look_size):
    """Save a train/val/test split as HDF5.

    Parameters
    ----------
    path : str — output .h5 file
    noisy_list : list of (256, 256) float32 arrays
    clean_list : list of (256, 256) float32 arrays
    source_list : list of str ('grd' or 'slc')
    """
    N = len(noisy_list)
    assert N == len(clean_list) == len(source_list)

    H, W = noisy_list[0].shape
    chunk = min(64, N)

    with h5py.File(path, "w") as f:
        ds_noisy = f.create_dataset(
            "noisy", shape=(N, 1, H, W), dtype="float32",
            chunks=(chunk, 1, H, W), compression="gzip", compression_opts=4,
        )
        ds_clean = f.create_dataset(
            "clean", shape=(N, 1, H, W), dtype="float32",
            chunks=(chunk, 1, H, W), compression="gzip", compression_opts=4,
        )
        # Store source as variable-length bytes
        dt = h5py.special_dtype(vlen=bytes)
        ds_source = f.create_dataset("source", shape=(N,), dtype=dt)

        for i in range(N):
            ds_noisy[i] = noisy_list[i][np.newaxis, :, :]
            ds_clean[i] = clean_list[i][np.newaxis, :, :]
            ds_source[i] = source_list[i].encode("utf-8")

        # Metadata
        f.attrs["n_samples"] = N
        f.attrs["patch_size"] = H
        f.attrs["data_type"] = "real_sentinel1"
        f.attrs["smooth_method"] = smooth_method
        f.attrs["look_size"] = look_size
        f.attrs["grd_path"] = GRD_PATH
        f.attrs["slc_dirs"] = "\n".join(SLC_DIRS)

    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  Saved {path} ({N} samples, {size_mb:.1f} MB)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate SAR despeckling dataset from real Sentinel-1 data")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: ./data)")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                        help="Root containing Sentinel-1 GRD/SLC source data "
                             "(default: P04_SAR_DATA_ROOT or REMI Lab NFS path)")
    parser.add_argument("--grd_path", type=str, default=None,
                        help="Override the single processed Sentinel-1 GRD .npz path")
    parser.add_argument("--slc_dir", action="append", default=None,
                        help="Override/add an SLC focused directory. Can be repeated; "
                             "if omitted, uses the canonical REMI Lab focused directories.")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--look_size", type=int, default=4,
                        help="Multi-look block size for SLC pseudo-GT (default: 4)")
    parser.add_argument("--smooth_method", type=str, default="multilook",
                        choices=["gaussian", "multilook"],
                        help="SLC pseudo-GT smoothing: 'multilook' is the lecture "
                             "default; 'gaussian' is the v9 ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: skip SLC, use only GRD non-overlap")
    args = parser.parse_args()

    global GRD_PATH, SLC_DIRS
    default_grd_path, default_slc_dirs = _resolve_default_paths(args.data_root)
    GRD_PATH = args.grd_path if args.grd_path else default_grd_path
    SLC_DIRS = args.slc_dir if args.slc_dir else default_slc_dirs

    if args.out_dir is None:
        args.out_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "data"))
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("=== Real Sentinel-1 SAR Despeckling Dataset Generation ===")
    print(f"  patch_size   = {args.patch_size}")
    print(f"  look_size    = {args.look_size}")
    print(f"  smooth       = {args.smooth_method}")
    print(f"  data_root    = {args.data_root}")
    print(f"  grd_path     = {GRD_PATH}")
    print(f"  slc_dirs     = {len(SLC_DIRS)}")
    print(f"  output       = {args.out_dir}")
    print()

    # ── Step 1: GRD patches ──
    print("[Step 1] Extracting GRD patches...")
    if args.smoke:
        # Smoke test: only non-overlapping GRD patches
        print("  (smoke mode: non-overlapping only)")
        grd_patches_raw = extract_grd_patches(
            patch_size=args.patch_size, overlap_stride=args.patch_size)
    else:
        grd_patches_raw = extract_grd_patches(
            patch_size=args.patch_size, overlap_stride=128)

    grd_patches = filter_patches(grd_patches_raw)
    print()

    # ── Step 2: SLC patches ──
    if args.smoke:
        print("[Step 2] Skipping SLC patches (smoke mode)")
        slc_patches = []
    else:
        print("[Step 2] Extracting SLC patches...")
        slc_patches_raw = extract_all_slc_patches(
            patch_size=args.patch_size, look_size=args.look_size,
            smooth_method=args.smooth_method)
        slc_patches = filter_patches(slc_patches_raw)
    print()

    # ── Step 3: Combine and split ──
    print("[Step 3] Combining and splitting...")

    # Build arrays with source labels
    all_noisy = []
    all_clean = []
    all_source = []

    for noisy, clean in grd_patches:
        all_noisy.append(noisy)
        all_clean.append(clean)
        all_source.append("grd")

    for noisy, clean in slc_patches:
        all_noisy.append(noisy)
        all_clean.append(clean)
        all_source.append("slc")

    N = len(all_noisy)
    print(f"  Total patches: {N} (GRD: {len(grd_patches)}, SLC: {len(slc_patches)})")

    if N == 0:
        print("  ERROR: No patches extracted. Check data paths.")
        return

    # Shuffle
    indices = rng.permutation(N)
    all_noisy = [all_noisy[i] for i in indices]
    all_clean = [all_clean[i] for i in indices]
    all_source = [all_source[i] for i in indices]

    # Split
    n_train = int(N * args.train_frac)
    n_val = int(N * args.val_frac)
    n_test = N - n_train - n_val

    # Ensure at least 1 sample per split
    if n_val < 1:
        n_val = 1
        n_train = N - n_val - max(1, n_test)
        n_test = N - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = N - n_val - n_test

    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")

    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, N),
    }

    # ── Step 4: Save HDF5 ──
    print()
    print("[Step 4] Saving HDF5 files...")
    for split_name, (start, end) in splits.items():
        fname = f"real_despeckling_{split_name}.h5"
        path = os.path.join(args.out_dir, fname)
        save_split(
            path,
            all_noisy[start:end],
            all_clean[start:end],
            all_source[start:end],
            smooth_method=args.smooth_method,
            look_size=args.look_size,
        )

    # ── Summary stats ──
    print()
    print("=== Dataset Summary ===")
    for split_name, (start, end) in splits.items():
        n = end - start
        sources = all_source[start:end]
        n_grd = sum(1 for s in sources if s == "grd")
        n_slc = sum(1 for s in sources if s == "slc")

        noisy_arr = np.array(all_noisy[start:end])
        clean_arr = np.array(all_clean[start:end])
        print(f"  {split_name:>5s}: {n:>5d} patches "
              f"(GRD={n_grd}, SLC={n_slc})  "
              f"noisy=[{noisy_arr.min():.3f}, {noisy_arr.max():.3f}]  "
              f"clean=[{clean_arr.min():.3f}, {clean_arr.max():.3f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
