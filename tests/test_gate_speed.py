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
    def test_known_gate_spacing_uses_ten_meters_even_if_json_differs(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 25, (100, 200), (300, 200)),
            gate("Gate 3", 50, (100, 300), (300, 300)),
        ], (WIDTH, HEIGHT), known_gate_spacing=True)
        self.assertEqual([item.station_m for item in estimator.gates], [0, 10, 20])
        self.assertEqual(estimator.metric_calibrator.as_dict()["gate_spacing_m"], 10)
        for frame, y in enumerate((80, 120, 180, 220), 1):
            estimator.update(7, (200, y), (frame - 1) * 0.4, frame)
        crossing = estimator.tracks[7].latest
        self.assertIsNotNone(crossing)
        self.assertAlmostEqual(crossing.distance_m, 10.0)
        self.assertAlmostEqual(crossing.kmh, 45.0, delta=0.1)

    def test_optional_gate_splits_ten_meter_interval_into_four_and_six(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 1.4", 8, (100, 140), (300, 140)),
            gate("Gate 2", 25, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT), known_gate_spacing=True)
        self.assertEqual([item.station_m for item in estimator.gates], [0, 4, 10])

    def test_sensitive_gate_projection_does_not_turn_jitter_into_high_speed(self):
        width, height = 1920, 1080

        def normalized(x, y):
            return {"x": x / (width - 1), "y": y / (height - 1)}

        gates = [
            {
                "gate": f"Gate {index + 1}", "station_m": index * 25,
                "lane": 2,
                "line": {
                    "L": normalized(500 + 30 * index, 400 + 9 * index),
                    "R": normalized(1500 - 30 * index, 400 + 9 * index),
                },
            }
            for index in range(3)
        ]
        for physical_kmh in (0, 3, 70):
            with self.subTest(physical_kmh=physical_kmh):
                estimator = GateSpeedEstimator(gates, (width, height), known_gate_spacing=True)
                inverse = np.linalg.inv(estimator.station_sections[0].transform)
                random = np.random.default_rng(4)
                estimates = []
                for frame in range(1, 61):
                    station = 5 + physical_kmh / 3.6 * frame / 60
                    pixel = inverse @ np.array([0.5, station, 1.0])
                    point_xy = pixel[:2] / pixel[2] + random.normal(0, 2, 2)
                    if frame == 35:
                        point_xy[1] += 6
                    estimates.append(estimator.update(
                        1, tuple(point_xy), frame / 60, frame
                    ))
                self.assertIsNotNone(estimates[1])
                self.assertGreater(sum(
                    estimate is not None and estimate.confidence == "GATE"
                    for estimate in estimates
                ), 20)
                if physical_kmh <= 3:
                    self.assertLess(max(
                        estimate.kmh for estimate in estimates[15:]
                        if estimate is not None
                    ), 15)
                else:
                    self.assertGreater(estimates[-1].kmh, 45)
                    self.assertLess(estimates[-1].kmh, 90)

    def test_after_crossing_uses_current_sections_pixel_resolution(self):
        width, height = 1920, 1080

        def normalized(x, y):
            return {"x": x / (width - 1), "y": y / (height - 1)}

        gates = [
            {
                "gate": f"Gate {index + 1}", "station_m": index * 10,
                "lane": 1,
                "line": {
                    "L": normalized(500, y),
                    "R": normalized(1500, y),
                },
            }
            for index, y in enumerate((400, 409, 450))
        ]
        estimator = GateSpeedEstimator(gates, (width, height), known_gate_spacing=True)
        estimate = None
        for frame, station in enumerate((5, 7, 9, 11), 1):
            section = estimator.station_sections[0 if station <= 10 else 1]
            pixel = np.linalg.inv(section.transform) @ np.array([0.5, station, 1.0])
            point_xy = tuple(pixel[:2] / pixel[2])
            estimate = estimator.update(1, point_xy, frame * 0.1, frame)
        self.assertIsNotNone(estimate)
        self.assertGreater(estimate.kmh, 0)
        self.assertEqual(estimate.confidence, "GATE")

    def test_adjacent_gate_mapping_has_smooth_slope_at_boundary(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (900, 100)),
            gate("Gate 2", 10, (250, 200), (750, 200)),
            gate("Gate 3", 20, (350, 250), (650, 250)),
        ], (WIDTH, HEIGHT), known_gate_spacing=True)
        before = estimator.road_position((500, 199.9))[0]
        at_gate = estimator.road_position((500, 200))[0]
        after = estimator.road_position((500, 200.1))[0]
        self.assertAlmostEqual(at_gate, 10, delta=0.01)
        self.assertAlmostEqual(
            (at_gate - before) / 0.1,
            (after - at_gate) / 0.1,
            delta=0.02,
        )

    def test_pixel_uncertainty_matches_gate_projection(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (900, 100)),
            gate("Gate 2", 10, (250, 200), (750, 200)),
        ], (WIDTH, HEIGHT), known_gate_spacing=True)
        matrix = estimator.regular_sections[0].transform
        position = (420.0, 155.0)

        def station(x, y):
            projected = matrix @ np.array([x, y, 1.0])
            return projected[1] / projected[2]

        expected = max(
            abs(station(position[0] + dx, position[1] + dy) - station(*position))
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
        )
        self.assertAlmostEqual(
            estimator.metric_calibrator.station_uncertainty(0, position), expected
        )

    def test_many_gate_spatial_shortlist_matches_full_projection(self):
        estimator = GateSpeedEstimator([
            gate(f"Gate {index + 1}", index * 10,
                 (100, 100 + index * 20), (300, 100 + index * 20))
            for index in range(20)
        ], (WIDTH, HEIGHT), stop_line=((100, 80), (300, 80)),
            known_gate_spacing=True)
        positions = [(x, y) for x in (80, 150, 250, 320)
                     for y in (70, 100, 111, 190, 291, 470, 510)]
        bounds = estimator.section_bounds
        estimator.section_bounds = [
            (float("-inf"), float("inf"), float("-inf"), float("inf"))
            for _ in bounds
        ]
        exhaustive = [estimator.road_position(position) for position in positions]
        estimator.section_bounds = bounds
        shortlisted = [estimator.road_position(position) for position in positions]
        for expected, actual in zip(exhaustive, shortlisted):
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertIsNotNone(actual)
                self.assertEqual(expected[2], actual[2])
                self.assertAlmostEqual(expected[0], actual[0])
                self.assertAlmostEqual(expected[1], actual[1])

    def test_speed_zone_forward_direction_shows_live_then_section_speed(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT), stop_line=((100, 50), (300, 50)),
            known_gate_spacing=True)
        readings = []
        states = []
        for frame, y in enumerate((20, 70, 120, 170, 220), 1):
            readings.append(estimator.update(7, (200, y), (frame - 1) * 0.4, frame))
            states.append(estimator.speed_zone_state(7))
            if frame < 5:
                self.assertIsNone(estimator.section_speed(7))
        self.assertEqual(states, ["before", "inside", "inside", "inside", "after"])
        self.assertIsNone(readings[0])
        self.assertIsNone(readings[-1])
        self.assertTrue(any(reading is not None for reading in readings[1:-1]))
        section = estimator.section_speed(7)
        self.assertIsNotNone(section)
        self.assertEqual((section.from_gate, section.to_gate), ("Gate 1", "Gate 2"))
        self.assertAlmostEqual(section.distance_m, 10)
        self.assertAlmostEqual(section.kmh, 45, delta=0.1)

    def test_speed_zone_reverse_direction_uses_same_boundaries(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT), stop_line=((100, 50), (300, 50)),
            known_gate_spacing=True)
        readings = []
        states = []
        for frame, y in enumerate((230, 180, 130, 80, 30), 1):
            readings.append(estimator.update(8, (200, y), (frame - 1) * 0.4, frame))
            states.append(estimator.speed_zone_state(8))
        self.assertEqual(states, ["before", "inside", "inside", "inside", "after"])
        self.assertIsNone(readings[0])
        self.assertIsNone(readings[-1])
        self.assertTrue(any(reading is not None for reading in readings[1:-1]))
        section = estimator.section_speed(8)
        self.assertIsNotNone(section)
        self.assertEqual((section.from_gate, section.to_gate), ("Gate 2", "Gate 1"))
        self.assertAlmostEqual(section.kmh, 45, delta=0.1)

    def test_tracking_gap_restarts_speed_zone_state(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT), stop_line=((100, 50), (300, 50)),
            known_gate_spacing=True)
        estimator.update(7, (200, 20), 0.0, 1)
        estimator.update(7, (200, 70), 0.4, 2)
        self.assertEqual(estimator.speed_zone_state(7), "inside")
        reading = estimator.update(7, (200, 120), 3.0, 12)
        self.assertEqual(estimator.speed_zone_state(7), "inside")
        self.assertIsNone(reading)
        self.assertIsNone(estimator.section_speed(7))

    def test_section_speed_stays_hidden_until_optional_gate_zone_exit(self):
        estimator = GateSpeedEstimator([
            gate("Gate 1", 0, (100, 100), (300, 100)),
            gate("Gate 1.4", 4, (100, 140), (300, 140)),
            gate("Gate 2", 10, (100, 200), (300, 200)),
        ], (WIDTH, HEIGHT), stop_line=((100, 50), (300, 50)),
            known_gate_spacing=True)
        for frame, y in enumerate((20, 70, 120, 160, 220), 1):
            estimator.update(7, (200, y), (frame - 1) * 0.4, frame)
            if frame < 5:
                self.assertIsNone(estimator.section_speed(7))
        section = estimator.section_speed(7)
        self.assertIsNotNone(section)
        self.assertAlmostEqual(section.distance_m, 10)
        self.assertEqual(
            (section.from_gate, section.to_gate),
            ("Gate 1", "Gate 2"),
        )

    def test_live_speed_limits_single_box_jump_in_dense_gate_area(self):
        width, height = 1920, 1080

        def normalized(x, y):
            return {"x": x / (width - 1), "y": y / (height - 1)}

        gates = [
            {
                "gate": f"Gate {index + 1}", "station_m": index * 10,
                "lane": 1,
                "line": {
                    "L": normalized(500 + 30 * index, 400 + 9 * index),
                    "R": normalized(1500 - 30 * index, 400 + 9 * index),
                },
            }
            for index in range(3)
        ]
        estimator = GateSpeedEstimator(gates, (width, height), known_gate_spacing=True)
        inverse = np.linalg.inv(estimator.regular_sections[0].transform)
        random = np.random.default_rng(7)
        speeds = []
        for frame in range(1, 91):
            station = 3 + 40 / 3.6 * frame / 60
            pixel = inverse @ np.array([0.5, station, 1.0])
            point_xy = pixel[:2] / pixel[2] + random.normal(0, 3, 2)
            if frame == 45:
                point_xy[1] += 20
            estimate = estimator.update(7, tuple(point_xy), frame / 60, frame)
            speeds.append(estimate.kmh if estimate else None)
        self.assertLess(max(speeds[20:]), 70)
        self.assertLess(max(
            abs(current - previous)
            for previous, current in zip(speeds[20:-1], speeds[21:])
        ), 4)
        self.assertGreater(speeds[-1], 20)

    def test_live_speed_follows_acceleration_and_braking(self):
        gates = [
            gate(f"Gate {index + 1}", index * 10,
                 (100, 100 + index * 100), (300, 100 + index * 100))
            for index in range(5)
        ]
        for start_kmh, finish_kmh in ((10, 60), (60, 10)):
            with self.subTest(start=start_kmh, finish=finish_kmh):
                estimator = GateSpeedEstimator(
                    gates, (WIDTH, HEIGHT), known_gate_spacing=True
                )
                random = np.random.default_rng(2)
                at_175 = None
                for frame in range(1, 121):
                    timestamp = frame / 60
                    station = (
                        start_kmh / 3.6 * min(timestamp, 1.0)
                        + finish_kmh / 3.6 * max(0.0, timestamp - 1.0)
                    )
                    point_xy = (
                        200 + random.normal(0, 1),
                        100 + 10 * station + random.normal(0, 1),
                    )
                    estimate = estimator.update(8, point_xy, timestamp, frame)
                    if frame == 105:
                        at_175 = estimate.kmh
                self.assertAlmostEqual(at_175, finish_kmh, delta=5)

    def test_five_fps_track_can_leave_provisional_state(self):
        estimator = GateSpeedEstimator([
            gate(f"Gate {index + 1}", index * 10,
                 (100, 100 + index * 100), (300, 100 + index * 100))
            for index in range(5)
        ], (WIDTH, HEIGHT), known_gate_spacing=True)
        estimate = None
        for frame in range(1, 9):
            timestamp = frame * 0.2
            estimate = estimator.update(
                7, (200, 100 + 40 / 3.6 * timestamp * 10), timestamp, frame
            )
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.confidence, "GATE")
        self.assertAlmostEqual(estimate.kmh, 40, delta=5)

    def test_local_speed_starts_after_entering_from_stop_line(self):
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
