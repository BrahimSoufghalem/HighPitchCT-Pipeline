import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from highpitchct.runtime import (
    GIB,
    KAGGLE_DEFAULT_MAX_DISK_GB,
    DiskBudgetGuard,
    default_data_root,
    ensure_writable_output,
    resolve_environment_profile,
)


class RuntimeTests(unittest.TestCase):
    def test_environment_override_wins(self):
        with patch.dict(os.environ, {"HPCT_DATA_ROOT": "/tmp/hpct-custom"}):
            self.assertEqual(default_data_root(), Path("/tmp/hpct-custom"))

    def test_writable_output_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new"
            self.assertEqual(ensure_writable_output(output), output.resolve())
            self.assertTrue(output.is_dir())
            self.assertFalse((output / ".hpct-write-test").exists())

    def test_kaggle_input_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            ensure_writable_output("/kaggle/input/forbidden-output")

    def test_auto_profile_switches_between_kaggle_18gb_and_full_unlimited(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "work"
            inc_dir = Path(tmp) / "inc"
            with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, clear=False):
                prof_k = resolve_environment_profile(
                    profile="auto", data_root=data_dir, incoming_root=inc_dir
                )
                self.assertEqual(prof_k.name, "kaggle")
                self.assertEqual(prof_k.max_disk_bytes, int(KAGGLE_DEFAULT_MAX_DISK_GB * GIB))
                self.assertTrue(prof_k.cleanup_archive_after_extract)

            with patch("highpitchct.runtime.is_kaggle", return_value=False):
                prof_f = resolve_environment_profile(profile="auto", data_root=data_dir)
                self.assertEqual(prof_f.name, "full")
                self.assertIsNone(prof_f.max_disk_bytes)
                self.assertFalse(prof_f.cleanup_archive_after_extract)

    def test_disk_budget_guard_blocks_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "existing.bin").write_bytes(b"x" * 600)
            guard = DiskBudgetGuard(root, max_disk_bytes=1000)
            ok1, _ = guard.try_reserve(300)
            self.assertTrue(ok1)
            ok2, _ = guard.try_reserve(200)
            self.assertFalse(ok2)
            guard.release(300)
            ok3, _ = guard.try_reserve(200)
            self.assertTrue(ok3)


if __name__ == "__main__":
    unittest.main()
