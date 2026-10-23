import tempfile
import unittest
import zipfile
from pathlib import Path

from highpitchct.safe_zip import safe_extract_zip, verify_zip


class SafeZipTests(unittest.TestCase):
    def test_valid_zip_extracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "ok.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("nested/a.dcm", b"DICOM")
            verify_zip(archive)
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


if __name__ == "__main__":
    unittest.main()
