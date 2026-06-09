from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from projects.p04_dncnn_sar.smoke_data import (
    copy_bundled_real_smoke_splits,
    write_synthetic_despeckling_splits,
)


def _read_data_type(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        return str(handle.attrs["data_type"])


class TestP04SyntheticSmokeData(unittest.TestCase):
    def test_write_clone_safe_smoke_splits_when_real_sentinel1_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            write_synthetic_despeckling_splits(out_dir, patch_size=32, seed=7)

            expected = {"train": 24, "val": 6, "test": 6}
            for split_name, expected_n in expected.items():
                path = out_dir / f"real_despeckling_{split_name}.h5"
                self.assertTrue(path.exists(), path)
                with h5py.File(path, "r") as handle:
                    noisy = handle["noisy"][:]
                    clean = handle["clean"][:]
                    source = handle["source"][:]

                    self.assertEqual(noisy.shape, (expected_n, 1, 32, 32))
                    self.assertEqual(clean.shape, (expected_n, 1, 32, 32))
                    self.assertEqual(noisy.dtype, np.float32)
                    self.assertEqual(clean.dtype, np.float32)
                    self.assertTrue(np.all((0.0 <= noisy) & (noisy <= 1.0)))
                    self.assertTrue(np.all((0.0 <= clean) & (clean <= 1.0)))
                    self.assertEqual(set(source), {b"synthetic_smoke"})
                    self.assertEqual(handle.attrs["data_type"], "synthetic_p04_smoke")

    def test_copy_bundled_real_smoke_splits_preserves_real_smoke_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            copy_bundled_real_smoke_splits(out_dir)

            self.assertEqual(_read_data_type(out_dir / "real_despeckling_train.h5"), "real_sentinel1_smoke")
            self.assertEqual(_read_data_type(out_dir / "real_despeckling_val.h5"), "real_sentinel1_smoke")
            self.assertEqual(_read_data_type(out_dir / "real_despeckling_test.h5"), "real_sentinel1_smoke")


if __name__ == "__main__":
    unittest.main()
