import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from highpitchct.manifest import SeriesSpec
from highpitchct.runtime import EnvironmentProfile
from highpitchct.tcia_client import (
    _download_once,
    download_manifest_with_profile,
    image_url,
)


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


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


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

    def test_kaggle_quota_stops_cleanly_while_full_downloads_all(self):
        zip_payload = _make_zip_bytes({"1-001.dcm": b"X" * 600})
        specs = [
            SeriesSpec("case_1", "1.2.1", "projection_full", expected_bytes=600, expected_files=1),
            SeriesSpec("case_2", "1.2.2", "projection_full", expected_bytes=600, expected_files=1),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kaggle_data = root / "kaggle_work"
            kaggle_tmp = root / "kaggle_tmp"
            kaggle_data.mkdir()
            kaggle_tmp.mkdir()

            kaggle_prof = EnvironmentProfile(
                name="kaggle",
                is_kaggle_env=True,
                data_root=kaggle_data,
                incoming_root=kaggle_tmp,
                max_disk_bytes=1000,  # Enough for case_1 (~600B + manifest), not case_2
                cleanup_archive_after_extract=True,
            )

            with patch(
                "urllib.request.urlopen",
                side_effect=lambda *a, **kw: FakeResponse(zip_payload, 200),
            ):
                summary_k = download_manifest_with_profile(specs, kaggle_prof, workers=1)

            self.assertEqual(summary_k["completed_series"], 1)
            self.assertEqual(summary_k["skipped_quota_series"], 1)
            self.assertTrue(summary_k["quota_stopped"])
            # Temporary zip in incoming_root was cleaned up after extraction!
            self.assertEqual(list(kaggle_tmp.rglob("*.zip")), [])

            # Re-running on Kaggle must recognize already_extracted without re-downloading!
            with patch("urllib.request.urlopen", side_effect=AssertionError("Should not download")):
                summary_k2 = download_manifest_with_profile([specs[0]], kaggle_prof, workers=1)
            self.assertEqual(summary_k2["completed_series"], 1)
            self.assertEqual(summary_k2["results"][0]["status"], "already_extracted")

            # Now run in FULL profile (max_disk_bytes=None): downloads ALL cases!
            full_data = root / "full_work"
            full_inc = full_data / "incoming"
            full_data.mkdir()
            full_inc.mkdir()
            full_prof = EnvironmentProfile(
                name="full",
                is_kaggle_env=False,
                data_root=full_data,
                incoming_root=full_inc,
                max_disk_bytes=None,
                cleanup_archive_after_extract=False,
            )
            with patch(
                "urllib.request.urlopen",
                side_effect=lambda *a, **kw: FakeResponse(zip_payload, 200),
            ):
                summary_f = download_manifest_with_profile(specs, full_prof, workers=2)
            self.assertEqual(summary_f["completed_series"], 2)
            self.assertEqual(summary_f["skipped_quota_series"], 0)


if __name__ == "__main__":
    unittest.main()
