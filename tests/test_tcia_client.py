from contextlib import contextmanager
import csv
import hashlib
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
    _download_stream_to_part,
    download_manifest_with_profile,
    image_url,
    make_http_session,
)


def _fake_stream(payload: bytes, status: int = 200, headers: dict[str, str] | None = None):
    @contextmanager
    def _ctx(url, req_headers, timeout, *, chunk_size=4 * 1024 * 1024):
        hdrs = {k.lower(): v for k, v in (headers or {}).items()}
        chunks = [payload[i : i + 7] for i in range(0, len(payload), 7)]
        yield status, hdrs, iter(chunks)

    return _ctx


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

    def test_make_http_session_configures_retry_pool(self):
        session = make_http_session(pool_maxsize=6, retries=4)
        try:
            adapter = session.get_adapter("https://services.cancerimagingarchive.net")
            self.assertEqual(adapter.max_retries.total, 4)
            self.assertIn(429, adapter.max_retries.status_forcelist)
            self.assertIn(503, adapter.max_retries.status_forcelist)
        finally:
            session.close()

    def test_http_206_appends_to_partial_file_and_hashes_full_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "series.zip.part"
            partial.write_bytes(b"first-")
            with patch(
                "highpitchct.tcia_client._open_http_stream",
                side_effect=_fake_stream(b"second", 206),
            ):
                size, resumed, sha = _download_stream_to_part("https://example.test", partial, 5)
            self.assertTrue(resumed)
            self.assertEqual(size, 12)
            self.assertEqual(partial.read_bytes(), b"first-second")
            self.assertEqual(sha, hashlib.sha256(b"first-second").hexdigest())

    def test_http_200_restarts_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "series.zip.part"
            partial.write_bytes(b"stale-")
            with patch(
                "highpitchct.tcia_client._open_http_stream",
                side_effect=_fake_stream(b"fresh", 200),
            ):
                size, resumed = _download_once("https://example.test", partial, 5)
            self.assertFalse(resumed)
            self.assertEqual(size, 5)
            self.assertEqual(partial.read_bytes(), b"fresh")

    def test_kaggle_quota_stops_cleanly_while_full_downloads_all_in_parallel(self):
        zip_payload = _make_zip_bytes({"1-001.dcm": b"X" * 600})
        expected_sha = hashlib.sha256(zip_payload).hexdigest()
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
                max_disk_bytes=1000,  # Enough for case_1 (~600B), not case_2
                cleanup_archive_after_extract=True,
            )

            with patch(
                "highpitchct.tcia_client._open_http_stream",
                side_effect=_fake_stream(zip_payload, 200),
            ):
                summary_k = download_manifest_with_profile(
                    specs, kaggle_prof, workers=4, show_progress=False
                )

            self.assertEqual(summary_k["completed_series"], 1)
            self.assertEqual(summary_k["skipped_quota_series"], 1)
            self.assertTrue(summary_k["quota_stopped"])
            self.assertEqual(summary_k["results"][0]["archive_sha256"], expected_sha)
            self.assertEqual(list(kaggle_tmp.rglob("*.zip")), [])

            report_csv = Path(summary_k["download_report_csv"])
            self.assertTrue(report_csv.is_file())
            with report_csv.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "downloaded")
            self.assertEqual(rows[1]["status"], "skipped_quota_reached")

            # Re-running on Kaggle must recognize already_extracted without re-downloading!
            with patch(
                "highpitchct.tcia_client._open_http_stream",
                side_effect=AssertionError("Should not download"),
            ):
                summary_k2 = download_manifest_with_profile(
                    [specs[0]], kaggle_prof, workers=4, show_progress=False
                )
            self.assertEqual(summary_k2["completed_series"], 1)
            self.assertEqual(summary_k2["results"][0]["status"], "already_extracted")

            # Now run in FULL profile (max_disk_bytes=None): downloads ALL cases concurrently!
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
                "highpitchct.tcia_client._open_http_stream",
                side_effect=_fake_stream(zip_payload, 200),
            ):
                summary_f = download_manifest_with_profile(
                    specs, full_prof, workers=4, show_progress=False
                )
            self.assertEqual(summary_f["completed_series"], 2)
            self.assertEqual(summary_f["skipped_quota_series"], 0)


if __name__ == "__main__":
    unittest.main()
