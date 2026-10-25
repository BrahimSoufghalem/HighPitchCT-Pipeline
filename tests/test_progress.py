import io
import unittest
from unittest.mock import patch

from highpitchct import progress as progress_mod
from highpitchct.progress import ProgressReporter


class ProgressReporterTests(unittest.TestCase):
    def test_fallback_progress_tracks_updates_rollbacks_and_total_adjustments(self):
        buf = io.StringIO()
        with patch.object(progress_mod, "_tqdm", None):
            with ProgressReporter(
                total=1000,
                desc="Downloading",
                unit="B",
                enabled=True,
                mininterval=0.0,
                stream=buf,
            ) as rep:
                rep.write("Starting series 1")
                rep.update(400)
                self.assertEqual(rep.n, 400)
                rep.update(-150)  # Retry rollback
                self.assertEqual(rep.n, 250)
                rep.update(550)
                self.assertEqual(rep.n, 800)
                rep.adjust_total(-200)  # Reconcile compressed stream size to 800
                self.assertEqual(rep.total, 800)

        out = buf.getvalue()
        self.assertIn("Starting series 1", out)
        self.assertIn("100.0%", out)

    def test_disabled_progress_produces_no_output(self):
        buf = io.StringIO()
        with ProgressReporter(total=500, desc="Quiet", enabled=False, stream=buf) as rep:
            rep.write("hidden")
            rep.update(250)
            rep.adjust_total(-100)
            self.assertEqual(rep.n, 250)
            self.assertEqual(rep.total, 400)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
