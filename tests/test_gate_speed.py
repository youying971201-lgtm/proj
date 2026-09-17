import sys
import unittest
from pathlib import Path


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

    def test_lane_change_does_not_report_speed(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100), lanes=2),
            gate("Gate 2", 10, (100, 200), (300, 200), lanes=2),
        ], (WIDTH, HEIGHT))
        estimator.update(9, (150, 80), 0.0, 1)
        estimator.update(9, (150, 120), 0.4, 2)
        estimator.update(9, (250, 180), 0.8, 3)
        self.assertIsNone(estimator.update(9, (250, 220), 1.2, 4))

    def test_tracking_gap_resets_pending_gate_pair(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT))
        estimator.update(11, (200, 80), 0.0, 1)
        estimator.update(11, (200, 120), 0.4, 2)
        estimator.update(11, (200, 180), 5.0, 12)
        self.assertIsNone(estimator.update(11, (200, 220), 5.4, 13))

    def test_long_live_pause_resets_pending_gate_pair(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT))
        estimator.update(12, (200, 80), 0.0, 1)
        estimator.update(12, (200, 120), 0.4, 2)
        estimator.update(12, (200, 180), 5.0, 3)
        self.assertIsNone(estimator.update(12, (200, 220), 5.4, 4))

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
