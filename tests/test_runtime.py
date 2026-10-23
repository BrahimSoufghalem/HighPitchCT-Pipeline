import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from highpitchct.runtime import default_data_root, ensure_writable_output


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


if __name__ == "__main__":
    unittest.main()
