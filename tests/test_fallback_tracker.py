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

    def test_formal_id_handoff_matches_each_recent_vehicle_once(self):
        tracker = FallbackVehicleTracker()
        temporary = tracker.assign(
            [(100, 100, 140, 140), (300, 100, 340, 140)], 1
        )
        tracker.assign([], 2)
        matches = tracker.claim_formal_ids(
            [(305, 105, 345, 145), (105, 105, 145, 145)],
            [8, 9], 2, [], temporary,
        )
        self.assertEqual(set(matches), {(temporary[1], 8), (temporary[0], 9)})
        self.assertEqual(tracker.tracks, {})

    def test_formal_id_handoff_rejects_distant_or_current_temp_box(self):
        tracker = FallbackVehicleTracker()
        temporary = tracker.assign([(100, 100, 140, 140)], 1)[0]
        tracker.assign([], 2)
        self.assertEqual(tracker.claim_formal_ids(
            [(400, 100, 440, 140)], [8], 2, [], [temporary]
        ), [])
        self.assertEqual(tracker.claim_formal_ids(
            [(105, 105, 145, 145)], [8], 2, [temporary], [temporary]
        ), [])
        self.assertIn(temporary, tracker.tracks)

    def test_temporary_id_recovers_recent_formal_speed_track(self):
        tracker = FallbackVehicleTracker()
        tracker.observe_formal_ids([(100, 100, 140, 140)], [42], 1)
        temporary = tracker.assign([(105, 105, 145, 145)], 2)[0]
        self.assertEqual(tracker.claim_temporary_ids(
            [(105, 105, 145, 145)], [temporary], 2, [], [42]
        ), [(42, temporary)])
        self.assertEqual(tracker.claim_temporary_ids(
            [(105, 105, 145, 145)], [temporary], 2, [42], [42]
        ), [])


if __name__ == "__main__":
    unittest.main()
