import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gate_speed import GateSpeedEstimator, segment_intersection


WIDTH = 1001
HEIGHT = 651


def point(x, y):
    return {"x": x / (WIDTH - 1), "y": y / (HEIGHT - 1)}


def gate(name, station, left, right, lanes=1):
    return {
        "gate": name,
        "station_m": station,
        "lane": lanes,
        "line": {"L": point(*left), "R": point(*right)},
    }


class GateSpeedTests(unittest.TestCase):
    def test_local_speed_before_and_across_stop_line(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT), stop_line=((100, 50), (300, 50)))
        self.assertEqual(estimator.station_sections[0].first.name, "Stop line")
        estimate = None
        for frame, y in enumerate((20, 40, 60, 80), 1):
            estimate = estimator.update(5, (200, y), (frame - 1) * 0.2, frame)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "local")
        self.assertAlmostEqual(estimate.kmh, 36.0)

    def test_vertical_gate_and_stop_line_use_horizontal_motion(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (100, 300)),
            gate("Gate 2", 10, (200, 100), (200, 300)),
        ], (WIDTH, HEIGHT), stop_line=((50, 100), (50, 300)))
        estimate = None
        for frame, x in enumerate((20, 40, 60, 80), 1):
            estimate = estimator.update(6, (x, 200), (frame - 1) * 0.2, frame)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.kmh, 36.0)

    def test_slanted_gate_ignores_motion_along_its_width(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 200)),
            gate("Gate 2", 10, (100, 200), (300, 300)),
        ], (WIDTH, HEIGHT))
        estimate = None
        for frame, x in enumerate((120, 160, 200, 240), 1):
            estimate = estimator.update(
                8, (x, 150 + 0.5 * (x - 100)), (frame - 1) * 0.2, frame
            )
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.kmh, 0.0)

    def test_perspective_mapping_recovers_uniform_road_motion(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (900, 100)),
            gate("Gate 2", 10, (300, 200), (700, 200)),
        ], (WIDTH, HEIGHT))
        inverse = np.linalg.inv(estimator.station_sections[0].transform)
        estimate = None
        for frame, station in enumerate((0, 2.5, 5, 7.5, 10), 1):
            road_point = np.array([0.25, station, 1.0])
            image_point = inverse @ road_point
            point_xy = tuple(image_point[:2] / image_point[2])
            estimate = estimator.update(10, point_xy, (frame - 1) * 0.2, frame)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "local")
        self.assertAlmostEqual(estimate.kmh, 45.0)

    def test_interpolated_crossing_gives_interval_average(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT))
        estimator.update(7, (200, 80), 0.0, 1)
        estimator.update(7, (200, 120), 0.4, 2)
        estimator.update(7, (200, 180), 0.8, 3)
        estimate = estimator.update(7, (200, 220), 1.2, 4)

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.duration_s, 0.8)
        self.assertAlmostEqual(estimate.distance_m, 10.0)
        self.assertAlmostEqual(estimate.kmh, 45.0)
        self.assertTrue(estimate.lane_adjusted)
        self.assertEqual((estimate.from_gate, estimate.to_gate), ("Gate 1", "Gate 2"))

    def test_finite_gate_rejects_crossing_outside_span(self):
        self.assertIsNone(segment_intersection(
            (400, 80), (400, 120), (100, 100), (300, 100)
        ))

    def test_curved_outer_lane_has_longer_estimated_path(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100), lanes=2),
            gate("Gate 2", 10, (100, 200), (300, 200), lanes=2),
        ], (WIDTH, HEIGHT), {
            "L": [point(100, 100), point(90, 150), point(100, 200)],
            "R": [point(300, 100), point(340, 150), point(300, 200)],
        })

        inner = estimator.lane_distance_ratio(0, 0.25)
        outer = estimator.lane_distance_ratio(0, 0.75)
        self.assertIsNotNone(inner)
        self.assertIsNotNone(outer)
        self.assertGreater(outer, inner)

    def test_perspective_distortion_falls_back_to_base_gate_distance(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 200), (900, 200)),
            gate("Gate 2", 10, (300, 180), (700, 180)),
        ], (WIDTH, HEIGHT), {
            "L": [point(100, 200), point(300, 180)],
            "R": [point(900, 200), point(700, 180)],
        })
        self.assertIsNone(estimator.lane_distance_ratio(0, 0.75))

        first = (700, 200)
        second = (600, 180)
        estimate = None
        for frame, (fraction, timestamp) in enumerate(
            [(-0.25, 0.0), (0.25, 0.2), (0.75, 0.6), (1.25, 0.8)], 1
        ):
            position = tuple(
                a + (b - a) * fraction for a, b in zip(first, second)
            )
            estimate = estimator.update(7, position, timestamp, frame)

        self.assertIsNotNone(estimate)
        self.assertFalse(estimate.lane_adjusted)
        self.assertAlmostEqual(estimate.distance_m, 10.0)
        self.assertAlmostEqual(estimate.kmh, 60.0)

    def test_lane_change_skips_gate_average_but_keeps_local_estimate(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100), lanes=2),
            gate("Gate 2", 10, (100, 200), (300, 200), lanes=2),
        ], (WIDTH, HEIGHT))
        estimator.update(9, (150, 80), 0.0, 1)
        estimator.update(9, (150, 120), 0.4, 2)
        estimator.update(9, (250, 180), 0.8, 3)
        estimate = estimator.update(9, (250, 220), 1.2, 4)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "local")
        self.assertIsNone(estimator.tracks[9].latest)

    def test_tracking_gap_resets_pending_gate_pair(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT))
        estimator.update(11, (200, 80), 0.0, 1)
        estimator.update(11, (200, 120), 0.4, 2)
        estimator.update(11, (200, 180), 5.0, 12)
        estimate = estimator.update(11, (200, 220), 5.4, 13)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "local")
        self.assertIsNone(estimator.tracks[11].latest)

    def test_long_live_pause_resets_pending_gate_pair(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT))
        estimator.update(12, (200, 80), 0.0, 1)
        estimator.update(12, (200, 120), 0.4, 2)
        estimator.update(12, (200, 180), 5.0, 3)
        estimate = estimator.update(12, (200, 220), 5.4, 4)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "local")
        self.assertIsNone(estimator.tracks[12].latest)

    def test_reverse_travel_uses_optional_four_meter_interval(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 1.4", 4, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT))
        estimator.update(13, (200, 220), 0.0, 1)
        estimator.update(13, (200, 180), 0.4, 2)
        estimator.update(13, (200, 120), 0.8, 3)
        estimate = estimator.update(13, (200, 80), 1.2, 4)

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.kmh, 18.0)
        self.assertEqual((estimate.from_gate, estimate.to_gate), ("Gate 1.4", "Gate 1"))


if __name__ == "__main__":
    unittest.main()
