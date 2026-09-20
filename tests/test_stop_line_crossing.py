import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from Car import Vehicle
from main import detect_crossing


class StopLineCrossingTests(unittest.TestCase):
    def test_vertical_stop_line_detects_entry(self):
        vehicle = Vehicle(1, (80, 200), 1, 1)
        config = {
            "line": ((100, 100), (100, 300)),
            "in_transition": (1, -1),
        }
        self.assertIsNone(detect_crossing(vehicle, (80, 200), config))
        self.assertEqual(detect_crossing(vehicle, (120, 200), config), "IN")

    def test_slanted_stop_line_uses_finite_intersection(self):
        config = {
            "line": ((100, 100), (300, 200)),
            "in_transition": (-1, 1),
        }
        vehicle = Vehicle(2, (200, 120), 1, 1)
        self.assertIsNone(detect_crossing(vehicle, (200, 120), config))
        self.assertEqual(detect_crossing(vehicle, (200, 180), config), "IN")

        outside = Vehicle(3, (400, 220), 1, 1)
        self.assertIsNone(detect_crossing(outside, (400, 220), config))
        self.assertIsNone(detect_crossing(outside, (400, 280), config))

    def test_crossing_counts_when_current_point_passes_endpoint(self):
        vehicle = Vehicle(4, (280, 80), 1, 1)
        config = {
            "line": ((100, 100), (300, 100)),
            "in_transition": (-1, 1),
        }
        self.assertIsNone(detect_crossing(vehicle, (280, 80), config))
        self.assertEqual(detect_crossing(vehicle, (320, 120), config), "IN")


if __name__ == "__main__":
    unittest.main()
