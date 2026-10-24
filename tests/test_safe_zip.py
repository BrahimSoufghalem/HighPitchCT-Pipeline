import tempfile
import unittest
import zipfile
from pathlib import Path

from highpitchct.runtime import DiskBudgetExceededError, DiskBudgetGuard
from highpitchct.safe_zip import safe_extract_zip, verify_zip


class SafeZipTests(unittest.TestCase):
    def test_valid_zip_extracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "ok.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("nested/a.dcm", b"DICOM")
            members, uncompressed = verify_zip(archive)
            self.assertEqual(members, 1)
            self.assertEqual(uncompressed, 5)
            count = safe_extract_zip(archive, root / "out")
            self.assertEqual(count, 1)
            self.assertEqual((root / "out/nested/a.dcm").read_bytes(), b"DICOM")

    def test_zip_slip_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escape.txt", b"no")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                safe_extract_zip(archive, root / "out")
            self.assertFalse((root / "escape.txt").exists())

    def test_pre_extraction_quota_blocks_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            archive = root / "big.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("slice.dcm", b"A" * 2048)
            guard = DiskBudgetGuard(data_root, max_disk_bytes=1024)
            with self.assertRaises(DiskBudgetExceededError):
                safe_extract_zip(archive, data_root / "extracted", budget_guard=guard)
            self.assertFalse((data_root / "extracted").exists())


if __name__ == "__main__":
    unittest.main()
