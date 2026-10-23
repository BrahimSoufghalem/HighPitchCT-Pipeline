import tempfile
import unittest
from pathlib import Path

from highpitchct.manifest import read_series_manifest


class ManifestTests(unittest.TestCase):
    def write(self, body: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "series.csv"
        path.write_text(body, encoding="utf-8")
        return path

    def test_disabled_example_is_ignored(self):
        path = self.write(
            "case_id,series_instance_uid,role,description,enabled\n"
            "case,1.2.3,unknown,example,false\n"
        )
        self.assertEqual(read_series_manifest(path), [])

    def test_enabled_row_is_parsed(self):
        path = self.write(
            "case_id,series_instance_uid,role,description,enabled\n"
            "case,1.2.840.1,projection_full,raw,true\n"
        )
        result = read_series_manifest(path)
        self.assertEqual(result[0].role, "projection_full")

    def test_duplicate_uid_is_rejected(self):
        path = self.write(
            "case_id,series_instance_uid,role,description,enabled\n"
            "a,1.2.3,unknown,x,true\n"
            "b,1.2.3,unknown,y,true\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_series_manifest(path)

    def test_invalid_uid_is_rejected(self):
        path = self.write(
            "case_id,series_instance_uid,role,description,enabled\n"
            "a,not-a-uid,unknown,x,true\n"
        )
        with self.assertRaisesRegex(ValueError, "invalid DICOM"):
            read_series_manifest(path)


if __name__ == "__main__":
    unittest.main()
