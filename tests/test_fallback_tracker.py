import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from main import FallbackVehicleTracker


class FallbackVehicleTrackerTests(unittest.TestCase):
    def test_multiple_boxes_keep_distinct_temporary_ids(self):
        tracker = FallbackVehicleTracker()
        first = tracker.assign([(100, 100, 140, 140), (300, 100, 340, 140)], 1)
        second = tracker.assign([(110, 110, 150, 150), (310, 110, 350, 150)], 2)
        self.assertEqual(first, second)
        self.assertEqual(len(set(second)), 2)
        self.assertTrue(all(track_id < 0 for track_id in second))

    def test_old_temporary_track_expires(self):
        tracker = FallbackVehicleTracker(max_gap_frames=2)
        first = tracker.assign([(100, 100, 140, 140)], 1)[0]
        later = tracker.assign([(100, 100, 140, 140)], 5)[0]
        self.assertNotEqual(first, later)


if __name__ == "__main__":
    unittest.main()
