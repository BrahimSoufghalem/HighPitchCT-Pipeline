import unittest
from pathlib import Path

from highpitchct.layout import safe_component, series_paths
from highpitchct.manifest import SeriesSpec


class LayoutTests(unittest.TestCase):
    def test_paths_are_deterministic(self):
        spec = SeriesSpec("pilot chest", "1.2.3", "projection_full")
        paths = series_paths("data", spec)
        self.assertEqual(paths.archive, Path("data/incoming/pilot_chest/projection_full/1.2.3.zip"))
        self.assertEqual(paths.extracted, Path("data/raw/tcia/pilot_chest/projection_full/1.2.3"))

    def test_empty_component_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_component("///")


if __name__ == "__main__":
    unittest.main()
