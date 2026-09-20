import sys
import unittest
from dataclasses import dataclass
from math import hypot
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lane_change import LaneChangeDetector, LaneLayout, build_equal_lane_layout


@dataclass(frozen=True)
class Gate:
    station_m: float


class CurvedBoundaryLayoutTests(unittest.TestCase):
    def setUp(self):
        self.gates = [Gate(0.0), Gate(10.0), Gate(20.0)]
        self.pair_paths = [
            (
                [(0, 100), (8, 80), (20, 60)],
                [(120, 100), (105, 80), (90, 60)],
            ),
            (
                [(20, 60), (32, 42), (45, 28)],
                [(90, 60), (82, 42), (75, 28)],
            ),
        ]

    def test_curved_dividers_are_equal_distance_between_edges_at_every_sample(self):
        layout = build_equal_lane_layout(self.gates, self.pair_paths, 3)
        self.assertTrue(layout.ready, layout.message)
        self.assertEqual(len(layout.display_polylines), 2)
        divider = layout.display_polylines[0]
        paired_samples = [
            pair for pair_paths in self.pair_paths
            for pair in zip(*pair_paths)
        ]
        # Adjacent pairs share one endpoint; the rendered polyline stores it only once.
        paired_samples.pop(3)
        for point, (left, right) in zip(divider, paired_samples):
            full_width = hypot(right[0] - left[0], right[1] - left[1])
            left_distance = hypot(point[0] - left[0], point[1] - left[1])
            self.assertAlmostEqual(left_distance / full_width, 1 / 3, places=6)

    def test_curved_point_projection_uses_piecewise_boundary_quads(self):
        layout = build_equal_lane_layout(self.gates, self.pair_paths, 3)
        left = self.pair_paths[0][0][1]
        right = self.pair_paths[0][1][1]
        quarter = (
            left[0] * 0.75 + right[0] * 0.25,
            left[1] * 0.75 + right[1] * 0.25,
        )
        self.assertAlmostEqual(layout.curved_across(quarter, 0), 0.25, places=2)

    def test_missing_boundary_pairs_disable_layout(self):
        layout = build_equal_lane_layout(self.gates, [], 3)
        self.assertFalse(layout.ready)

    def test_one_lane_layout_keeps_outer_boundary_detection_enabled(self):
        layout = build_equal_lane_layout(self.gates, self.pair_paths, 1)
        self.assertTrue(layout.ready, layout.message)
        self.assertEqual(layout.lane_count, 1)
        self.assertEqual(layout.as_dict()["separator_count"], 0)
        self.assertEqual(layout.display_polylines, ())
        self.assertEqual(layout.separators_at_gates, ((), (), ()))

    def test_full_divider_runs_from_stop_line_to_boundary_endpoints(self):
        boundaries = {
            "L": [(0, 125), (4, 112), (20, 60), (45, 28), (55, 12)],
            "R": [(130, 125), (124, 110), (90, 60), (75, 28), (69, 12)],
        }
        layout = build_equal_lane_layout(
            self.gates,
            self.pair_paths,
            2,
            boundary_paths=boundaries,
        )
        self.assertTrue(layout.ready, layout.message)
        self.assertEqual(len(layout.display_polylines), 1)
        divider = layout.display_polylines[0]
        self.assertEqual(divider[0], (65.0, 125.0))
        self.assertAlmostEqual(divider[-1][0], 62.0)
        self.assertAlmostEqual(divider[-1][1], 12.0)
        self.assertLess(self.pair_paths[0][0][0][1], divider[0][1])
        self.assertGreater(self.pair_paths[-1][0][-1][1], divider[-1][1])
        self.assertIn("停止線", layout.message)

    def test_full_boundary_projection_works_beyond_first_and_last_gate(self):
        boundaries = {
            "L": [(0, 125), (20, 60), (55, 12)],
            "R": [(130, 125), (90, 60), (69, 12)],
        }
        layout = build_equal_lane_layout(
            self.gates,
            self.pair_paths,
            2,
            boundary_paths=boundaries,
        )
        for left, right in (
            (layout.pair_paths[0][0][1], layout.pair_paths[0][1][1]),
            (layout.pair_paths[0][0][-2], layout.pair_paths[0][1][-2]),
        ):
            quarter = (
                left[0] * 0.75 + right[0] * 0.25,
                left[1] * 0.75 + right[1] * 0.25,
            )
            self.assertAlmostEqual(layout.curved_across(quarter, 0), 0.25, places=2)

    def test_divider_extrapolates_from_stop_line_to_screen_edge(self):
        boundaries = {
            "L": [(0, 125), (4, 112), (20, 60), (55, 12)],
            "R": [(130, 125), (124, 110), (90, 60), (69, 12)],
        }
        layout = build_equal_lane_layout(
            self.gates,
            self.pair_paths,
            2,
            boundary_paths=boundaries,
            frame_size=(200, 200),
        )
        self.assertTrue(layout.ready, layout.message)
        divider = layout.display_polylines[0]
        # The generated divider begins on an image edge, passes through the
        # stop-line midpoint, and continues to the far boundary endpoint.
        self.assertTrue(
            abs(divider[0][0]) < 1e-6
            or abs(divider[0][0] - 199) < 1e-6
            or abs(divider[0][1]) < 1e-6
            or abs(divider[0][1] - 199) < 1e-6
        )
        self.assertTrue(any(
            abs(point[0] - 65.0) < 1e-6 and abs(point[1] - 125.0) < 1e-6
            for point in divider
        ))
        self.assertAlmostEqual(divider[-1][0], 62.0)
        self.assertAlmostEqual(divider[-1][1], 12.0)
        self.assertIn("畫面邊緣", layout.message)

    def test_displayed_outer_boundaries_use_the_same_extended_geometry(self):
        boundaries = {
            "L": [(0, 125), (4, 112), (20, 60), (55, 12)],
            "R": [(130, 125), (124, 110), (90, 60), (69, 12)],
        }
        layout = build_equal_lane_layout(
            self.gates,
            self.pair_paths,
            2,
            boundary_paths=boundaries,
            frame_size=(200, 200),
        )
        self.assertEqual(layout.display_boundaries, layout.pair_paths[0])
        self.assertEqual(layout.as_dict()["outer_boundary_count"], 2)
        left, right = layout.display_boundaries
        self.assertGreater(len(left), len(boundaries["L"]))
        self.assertGreater(len(right), len(boundaries["R"]))


class LaneChangeDetectorTests(unittest.TestCase):
    def setUp(self):
        self.layout = LaneLayout(
            True,
            3,
            separators_at_gates=((1 / 3, 2 / 3),) * 3,
            gate_stations=(0.0, 10.0, 20.0),
            message="ready",
        )
        self.detector = LaneChangeDetector(
            self.layout,
            initial_confirm_frames=2,
            change_confirm_frames=3,
            changed_label_frames=2,
        )

    def update(self, across, frame, track_id=7):
        return self.detector.update(track_id, (5.0, across, 0), frame, frame / 10)

    def establish_lane_one(self):
        self.update(0.15, 1)
        return self.update(0.15, 2)

    def test_boundary_jitter_does_not_trigger(self):
        self.establish_lane_one()
        for frame, across in enumerate((0.34, 0.32, 0.35, 0.31), start=3):
            result = self.update(across, frame)
        self.assertEqual(result.lane_number, 1)
        self.assertEqual(self.detector.as_dict()["total"], 0)

    def test_confirms_right_lane_change_once(self):
        self.establish_lane_one()
        results = [self.update(0.48, frame) for frame in (3, 4, 5)]
        self.assertIsNone(results[1].event)
        self.assertEqual(results[2].event.direction, "RIGHT")
        self.assertEqual(results[2].event.from_lane, 1)
        self.assertEqual(results[2].event.to_lane, 2)
        self.update(0.48, 6)
        self.assertEqual(self.detector.as_dict()["total"], 1)

    def test_two_lane_right_change_uses_outer_edge_for_destination_width(self):
        layout = LaneLayout(
            True,
            2,
            separators_at_gates=((0.5,),) * 3,
            gate_stations=(0.0, 10.0, 20.0),
            message="ready",
        )
        detector = LaneChangeDetector(
            layout,
            initial_confirm_frames=2,
            change_confirm_frames=3,
            changed_label_frames=2,
        )
        detector.update(12, (5.0, 0.25, 0), 1, 0.1)
        detector.update(12, (5.0, 0.25, 0), 2, 0.2)
        results = [
            detector.update(12, (5.0, 0.60, 0), frame, frame / 10)
            for frame in (3, 4, 5)
        ]
        event = results[-1].event
        self.assertIsNotNone(event)
        self.assertFalse(event.outside)
        self.assertEqual(event.direction, "RIGHT")
        self.assertEqual(event.from_lane, 1)
        self.assertEqual(event.to_lane, 2)
        self.assertEqual(results[-1].overlay_label, "CHANGED-R 1>2")

    def test_outside_road_is_not_assigned(self):
        result = self.detector.update(7, (5.0, 1.3, 0), 1)
        self.assertEqual(result.state, "OUTSIDE")
        self.assertIsNone(result.lane_number)

    def test_crossing_left_outer_boundary_counts_as_lane_change(self):
        self.establish_lane_one()
        results = [self.update(-0.05, frame) for frame in (3, 4, 5)]
        event = results[-1].event
        self.assertTrue(event.outside)
        self.assertEqual(event.direction, "LEFT")
        self.assertEqual(event.from_lane, 1)
        self.assertEqual(event.to_lane, 0)
        self.assertEqual(results[-1].overlay_label, "CHANGED-L 1>OUT")
        self.assertEqual(self.detector.as_dict()["left"], 1)

    def test_reentering_after_outer_crossing_does_not_count_again(self):
        self.establish_lane_one()
        for frame in (3, 4, 5):
            self.update(-0.05, frame)
        for frame in (6, 7):
            result = self.update(0.15, frame)
        self.assertEqual(result.lane_number, 1)
        self.assertEqual(self.detector.as_dict()["total"], 1)

    def test_crossing_right_outer_boundary_counts_as_lane_change(self):
        self.update(0.85, 1)
        self.update(0.85, 2)
        results = [self.update(1.05, frame) for frame in (3, 4, 5)]
        event = results[-1].event
        self.assertTrue(event.outside)
        self.assertEqual(event.direction, "RIGHT")
        self.assertEqual(event.from_lane, 3)
        self.assertEqual(event.to_lane, 4)
        self.assertEqual(results[-1].overlay_label, "CHANGED-R 3>OUT")
        self.assertEqual(self.detector.as_dict()["right"], 1)

    def test_one_lane_crossing_right_outer_boundary_counts_as_lane_change(self):
        layout = LaneLayout(
            True,
            1,
            separators_at_gates=((), (), ()),
            gate_stations=(0.0, 10.0, 20.0),
            message="ready",
        )
        detector = LaneChangeDetector(
            layout,
            initial_confirm_frames=2,
            change_confirm_frames=3,
            changed_label_frames=2,
        )
        detector.update(11, (5.0, 0.5, 0), 1, 0.1)
        detector.update(11, (5.0, 0.5, 0), 2, 0.2)
        results = [
            detector.update(11, (5.0, 1.10, 0), frame, frame / 10)
            for frame in (3, 4, 5)
        ]
        event = results[-1].event
        self.assertIsNotNone(event)
        self.assertTrue(event.outside)
        self.assertEqual(event.direction, "RIGHT")
        self.assertEqual(event.from_lane, 1)
        self.assertEqual(event.to_lane, 2)
        self.assertEqual(results[-1].overlay_label, "CHANGED-R 1>OUT")
        self.assertEqual(detector.as_dict()["right"], 1)

    def test_one_lane_track_starting_outside_can_enter_without_error_or_event(self):
        layout = LaneLayout(
            True,
            1,
            separators_at_gates=((), (), ()),
            gate_stations=(0.0, 10.0, 20.0),
            message="ready",
        )
        for outside_across, side in ((-0.10, "LEFT"), (1.10, "RIGHT")):
            with self.subTest(side=side):
                detector = LaneChangeDetector(
                    layout,
                    initial_confirm_frames=2,
                    change_confirm_frames=3,
                )
                outside_results = [
                    detector.update(21, (5.0, outside_across, 0), frame, frame / 10)
                    for frame in (1, 2, 3)
                ]
                self.assertTrue(all(result.state == "OUTSIDE" for result in outside_results))
                self.assertTrue(all(result.lane_number is None for result in outside_results))
                self.assertEqual(outside_results[-1].overlay_label, f"OUT-{side[0]}")
                detector.update(21, (5.0, 0.50, 0), 4, 0.4)
                inside = detector.update(21, (5.0, 0.50, 0), 5, 0.5)
                self.assertEqual(inside.lane_number, 1)
                self.assertIsNone(inside.event)
                self.assertEqual(detector.as_dict()["total"], 0)

    def test_all_lane_counts_and_adjacent_directions_are_index_safe(self):
        for lane_count in range(1, 7):
            separators = tuple(index / lane_count for index in range(1, lane_count))
            layout = LaneLayout(
                True,
                lane_count,
                separators_at_gates=(separators,) * 3,
                gate_stations=(0.0, 10.0, 20.0),
                message="ready",
            )
            for from_lane in range(1, lane_count + 1):
                for direction in ("LEFT", "RIGHT"):
                    with self.subTest(
                        lane_count=lane_count,
                        from_lane=from_lane,
                        direction=direction,
                    ):
                        detector = LaneChangeDetector(
                            layout,
                            initial_confirm_frames=2,
                            change_confirm_frames=3,
                        )
                        start = (from_lane - 0.5) / lane_count
                        if direction == "LEFT":
                            target_lane = from_lane - 1
                            target = (
                                -0.10 if target_lane == 0
                                else (target_lane - 0.5) / lane_count
                            )
                        else:
                            target_lane = from_lane + 1
                            target = (
                                1.10 if target_lane > lane_count
                                else (target_lane - 0.5) / lane_count
                            )
                        detector.update(31, (5.0, start, 0), 1, 0.1)
                        detector.update(31, (5.0, start, 0), 2, 0.2)
                        results = [
                            detector.update(31, (5.0, target, 0), frame, frame / 10)
                            for frame in (3, 4, 5)
                        ]
                        event = results[-1].event
                        self.assertIsNotNone(event)
                        self.assertEqual(event.from_lane, from_lane)
                        self.assertEqual(event.to_lane, target_lane)
                        self.assertEqual(event.direction, direction)


if __name__ == "__main__":
    unittest.main()
