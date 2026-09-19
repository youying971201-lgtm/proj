import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gate_speed import GateSpeedEstimator
from main import vehicle_ground_contact
from visual_scale import VEHICLE_PRIORS, VisualScaleCalibrator


def synthetic_road(nominal_step=10, gate_slant=0.0):
    """用已知相機與三維車體生成影像，不借用演算法的目標速度。"""
    road_width = 7.0
    focal = 1600.0
    intrinsics = np.array([
        [focal, 0, 959.5], [0, focal, 539.5], [0, 0, 1.0],
    ])
    camera_position = np.array([3.5, -15.0, 8.0])
    target = np.array([3.5, 25.0, 0.0])
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    translation = -rotation @ camera_position
    projection = intrinsics @ np.column_stack([rotation, translation])

    def project(across, station, elevation=0.0):
        pixel = projection @ np.array([
            across * road_width, station, elevation, 1.0,
        ])
        return tuple(pixel[:2] / pixel[2])

    def normalized(across, station):
        x, y = project(across, station)
        return {"x": x / 1919, "y": y / 1079}

    gates = [
        {
            "gate": f"Gate {index + 1}", "lane": 2,
            "station_m": index * nominal_step,
            "line": {
                "L": normalized(0, index * 10 - gate_slant / 2),
                "R": normalized(1, index * 10 + gate_slant / 2),
            },
        }
        for index in range(3)
    ]
    estimator = GateSpeedEstimator(gates, (1920, 1080))
    calibrator = VisualScaleCalibrator(estimator.station_sections, (1920, 1080), 2)
    return estimator, calibrator, project, projection


def projected_car_box(projection, across, station, size, road_width=7.0):
    length, width, height = size
    pixels = []
    for lateral in (-width / 2, width / 2):
        for along in (-length / 2, length / 2):
            for elevation in (0, height):
                point = projection @ np.array([
                    across * road_width + lateral,
                    station + along, elevation, 1.0,
                ])
                pixels.append(point[:2] / point[2])
    pixels = np.asarray(pixels)
    return (
        float(pixels[:, 0].min()), float(pixels[:, 1].min()),
        float(pixels[:, 0].max()), float(pixels[:, 1].max()),
    )


class VisualScaleTests(unittest.TestCase):
    def test_multiple_3d_vehicle_boxes_recover_unmeasured_gate_length(self):
        estimator, calibrator, project, projection = synthetic_road()
        self.assertFalse(calibrator.ready)
        for track_id, (across, station) in enumerate((
            (0.25, 3), (0.65, 4), (0.4, 6), (0.7, 8),
            (0.3, 12), (0.6, 16), (0.45, 5), (0.55, 14), (0.35, 7),
        ), 1):
            box = projected_car_box(
                projection, across, station, VEHICLE_PRIORS[2]
            )
            calibrator.observe(track_id, box, 2, 0.9, track_id * 15)
        self.assertTrue(calibrator.ready)
        self.assertAlmostEqual(calibrator.metric_station(10), 10, delta=1.5)
        self.assertIn(calibrator.confidence, ("LOW", "MED"))

        estimator.metric_calibrator = calibrator
        estimate = None
        for frame, station in enumerate((3, 4, 5, 6, 7, 8), 1):
            estimate = estimator.update(
                99, project(0.5, station), (frame - 1) * 0.1, frame
            )
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "visual")
        self.assertAlmostEqual(estimate.kmh, 36, delta=6)

    def test_single_vehicle_clears_estimate_after_stable_track(self):
        estimator, calibrator, project, projection = synthetic_road()
        for frame in range(1, 20):
            box = projected_car_box(
                projection, 0.5, 3 + frame * 0.1, VEHICLE_PRIORS[2]
            )
            calibrator.observe(1, box, 2, 0.9, frame * 15)
        self.assertFalse(calibrator.ready)
        estimator.metric_calibrator = calibrator
        labels = []
        for frame, station in enumerate((3, 4, 5, 6, 7), 1):
            estimate = estimator.update(
                1, project(0.5, station), (frame - 1) * 0.1, frame
            )
            labels.append(estimate.confidence if estimate else None)
        self.assertIsNotNone(estimate)
        self.assertGreater(estimate.kmh, 0)
        self.assertEqual(labels[:4], [None, "PROVISIONAL", "PROVISIONAL", "TRACKED"])
        self.assertEqual(estimate.confidence, "TRACKED")
        self.assertFalse(calibrator.ready)

    def test_large_lane_count_does_not_invert_fit_bounds(self):
        estimator, _, _, projection = synthetic_road()
        calibrator = VisualScaleCalibrator(
            estimator.station_sections, (1920, 1080), 20
        )
        for track_id, station in enumerate((3, 4, 5, 6, 7, 8, 9, 12, 15), 1):
            box = projected_car_box(
                projection, 0.5, station, VEHICLE_PRIORS[2]
            )
            calibrator.observe(track_id, box, 2, 0.9, track_id * 15)
        self.assertFalse(calibrator.ready)

    def test_prefilled_station_units_do_not_fix_physical_gate_distance(self):
        _, calibrator, _, projection = synthetic_road(nominal_step=20)
        for track_id, (across, station) in enumerate((
            (0.25, 3), (0.65, 4), (0.4, 6), (0.7, 8),
            (0.3, 12), (0.6, 16), (0.45, 5), (0.55, 14), (0.35, 7),
        ), 1):
            box = projected_car_box(
                projection, across, station, VEHICLE_PRIORS[2]
            )
            calibrator.observe(track_id, box, 2, 0.9, track_id * 15)
        self.assertTrue(calibrator.ready)
        self.assertAlmostEqual(calibrator.metric_station(20), 10, delta=1.5)

    def test_provisional_speed_ignores_prefilled_station_unit(self):
        speeds = []
        for nominal_step in (10, 20):
            estimator, calibrator, project, _ = synthetic_road(
                nominal_step=nominal_step
            )
            estimator.metric_calibrator = calibrator
            first = estimator.update(1, project(0.5, 3), 0.0, 1)
            second = estimator.update(1, project(0.5, 4), 0.1, 2)
            self.assertIsNone(first)
            self.assertIsNotNone(second)
            self.assertEqual(second.confidence, "PROVISIONAL")
            speeds.append(second.kmh)
        self.assertAlmostEqual(speeds[0], speeds[1], delta=0.01)

    def test_vehicle_size_and_box_noise_keep_scale_within_coarse_range(self):
        _, calibrator, _, projection = synthetic_road()
        random = np.random.default_rng(1)
        for track_id in range(1, 14):
            across = random.uniform(0.2, 0.8)
            station = random.uniform(3, 17)
            sizes = tuple(
                nominal * factor
                for nominal, factor in zip(
                    VEHICLE_PRIORS[2], random.normal(1, 0.12, 3)
                )
            )
            box = np.asarray(projected_car_box(
                projection, across, station, sizes
            )) + random.normal(0, 2, 4)
            calibrator.observe(track_id, box, 2, 0.9, track_id * 15)
        self.assertTrue(calibrator.ready)
        self.assertAlmostEqual(calibrator.metric_station(10), 10, delta=2.5)

    def test_slanted_gates_still_fit_orthogonal_vehicle_body(self):
        _, calibrator, _, projection = synthetic_road(gate_slant=2.0)
        for track_id, (across, station) in enumerate((
            (0.25, 3), (0.65, 4), (0.4, 6), (0.7, 8),
            (0.3, 12), (0.6, 16), (0.45, 5), (0.55, 14), (0.35, 7),
        ), 1):
            box = projected_car_box(
                projection, across, station, VEHICLE_PRIORS[2]
            )
            calibrator.observe(track_id, box, 2, 0.9, track_id * 15)
        self.assertTrue(calibrator.ready)
        self.assertAlmostEqual(calibrator.metric_station(10), 10, delta=2)

    def test_one_bad_ground_point_does_not_double_visual_speed(self):
        estimator, calibrator, project, _ = synthetic_road()
        calibrator.road_width_m = 7.0
        calibrator.focal_px = 1600.0
        calibrator.relative_uncertainty = 0.2
        calibrator.revision = 1
        estimator.metric_calibrator = calibrator
        estimate = None
        for frame, station in enumerate((3, 4, 5, 12, 7, 8), 1):
            estimate = estimator.update(
                1, project(0.5, station), (frame - 1) * 0.1, frame
            )
            if frame == 4:
                self.assertIsNotNone(estimate)
                self.assertAlmostEqual(estimate.kmh, 36, delta=2)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.kmh, 36, delta=2)

    def test_single_ground_point_jump_at_60_fps_keeps_track(self):
        estimator, calibrator, project, _ = synthetic_road()
        calibrator.road_width_m = 7.0
        calibrator.focal_px = 1600.0
        calibrator.relative_uncertainty = 0.2
        estimator.metric_calibrator = calibrator
        for frame in range(1, 61):
            station = 2 + 10 * frame / 60 + (0.7 if frame == 35 else 0)
            estimate = estimator.update(
                1, project(0.5, station), frame / 60, frame
            )
            if frame in (35, 36, 40):
                self.assertIsNotNone(estimate)
                self.assertAlmostEqual(estimate.kmh, 36, delta=2)
        self.assertEqual(estimator.tracks[1].rejected_points, 0)

    def test_provisional_scale_also_rejects_ground_point_jump(self):
        estimator, calibrator, project, _ = synthetic_road()
        estimator.metric_calibrator = calibrator
        before_jump = None
        for frame in range(1, 61):
            station = 2 + 10 * frame / 60 + (0.7 if frame == 35 else 0)
            estimate = estimator.update(
                1, project(0.5, station), frame / 60, frame
            )
            if frame == 34:
                before_jump = estimate.kmh
            if frame in (35, 36, 40):
                self.assertIsNotNone(estimate)
                self.assertAlmostEqual(estimate.kmh, before_jump, delta=3)
        self.assertFalse(calibrator.ready)
        self.assertEqual(estimator.tracks[1].rejected_points, 0)

    def test_fast_vehicle_with_pixel_jitter_reaches_stable_speed(self):
        estimator, calibrator, project, _ = synthetic_road()
        calibrator.road_width_m = 7.0
        calibrator.focal_px = 1600.0
        calibrator.relative_uncertainty = 0.2
        estimator.metric_calibrator = calibrator
        random = np.random.default_rng(4)
        labels = []
        for frame in range(1, 61):
            x, y = project(0.5, 3 + 70 / 3.6 * frame / 60)
            point = (x + random.normal(0, 2), y + random.normal(0, 2))
            estimate = estimator.update(1, point, frame / 60, frame)
            labels.append(estimate.confidence if estimate else None)
        self.assertGreaterEqual(sum(label == "LOW" for label in labels), 45)
        self.assertEqual(labels[-1], "LOW")
        self.assertAlmostEqual(estimate.kmh, 70, delta=15)

    def test_straight_road_has_no_lane_length_difference(self):
        estimator, calibrator, _, _ = synthetic_road()
        calibrator.road_width_m = 7.0
        calibrator.focal_px = 1600.0
        for index, paths in enumerate(estimator.pair_paths):
            ratio = calibrator.lane_ratio(index, *paths, 0.8)
            self.assertIsNotNone(ratio)
            self.assertAlmostEqual(ratio, 1.0, delta=0.01)

    def test_ground_contact_prefers_mask_bottom_edge(self):
        box = (100, 100, 200, 300)
        polygon = [(130, 150), (170, 150), (180, 290), (140, 298)]
        self.assertEqual(vehicle_ground_contact(box, polygon), (160, 298))
        self.assertEqual(vehicle_ground_contact(box), (150, 300))


if __name__ == "__main__":
    unittest.main()
