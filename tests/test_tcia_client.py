import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from highpitchct.tcia_client import _download_once, image_url


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int):
        super().__init__(payload)
        self.status = status

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TciaClientTests(unittest.TestCase):
    def test_url_encodes_uid(self):
        url = image_url("1.2.840.123", "https://example.test/api/")
        self.assertEqual(
            url,
            "https://example.test/api/getImage?SeriesInstanceUID=1.2.840.123",
        )

    def test_http_206_appends_to_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "series.zip.part"
            partial.write_bytes(b"first-")
            response = FakeResponse(b"second", 206)
            with patch("urllib.request.urlopen", return_value=response):
                size, resumed = _download_once("https://example.test", partial, 5)
            self.assertTrue(resumed)
            self.assertEqual(size, 12)
            self.assertEqual(partial.read_bytes(), b"first-second")

    def test_http_200_restarts_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "series.zip.part"
            partial.write_bytes(b"stale-")
            response = FakeResponse(b"fresh", 200)
            with patch("urllib.request.urlopen", return_value=response):
                size, resumed = _download_once("https://example.test", partial, 5)
            self.assertFalse(resumed)
            self.assertEqual(size, 5)
            self.assertEqual(partial.read_bytes(), b"fresh")


if __name__ == "__main__":
    unittest.main()
