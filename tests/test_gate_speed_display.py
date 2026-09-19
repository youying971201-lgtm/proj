import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import main


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeCapture:
    def __init__(self, frame_count=6):
        self.position = 0
        self.frames = [np.zeros((650, 1000, 3), dtype=np.uint8) for _ in range(frame_count)]

    def isOpened(self):
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def get(self, property_id):
        return {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_POS_FRAMES: self.position,
            cv2.CAP_PROP_POS_MSEC: self.position * 100.0,
        }.get(property_id, 0)

    def set(self, property_id, value):
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)

    def release(self):
        pass


class FakeModel:
    def __init__(self, *_args, **_kwargs):
        self.frame_index = 0

    def track(self, *_args, **_kwargs):
        bottom = [80, 110, 140, 170, 200, 230][self.frame_index]
        self.frame_index += 1
        boxes = type("Boxes", (), {
            "xyxy": FakeTensor([[180, bottom - 40, 220, bottom]]),
            "id": FakeTensor([42]),
            "cls": FakeTensor([2]),
            "conf": FakeTensor([0.9]),
        })()
        return [type("Result", (), {"boxes": boxes})()]


class GateSpeedDisplayTests(unittest.TestCase):
    def test_live_speed_is_only_inside_zone_and_section_only_after_exit(self):
        class CrossingModel:
            frame_number = 0

            def __init__(self, *_args, **_kwargs):
                pass

            def track(self, *_args, **_kwargs):
                index = self.__class__.frame_number
                bottom = (0, 15, 30, 45, 70, 85, 100, 120, 170, 220, 250)[index]
                self.__class__.frame_number += 1
                boxes = type("Boxes", (), {
                    "xyxy": FakeTensor([[180, bottom - 40, 220, bottom]]),
                    "id": FakeTensor([42]) if index >= 5 and index != 6 else None,
                    "cls": FakeTensor([2]),
                    "conf": FakeTensor([0.9]),
                })()
                return [type("Result", (), {"boxes": boxes})()]

        saved_state = dict(main.app.state._state)
        saved_status = dict(main.runtime_status)
        text_calls = []
        original_put_text = cv2.putText

        def collect_text(frame, value, origin, *args):
            text_calls.append((CrossingModel.frame_number, value, origin, args[2]))
            return original_put_text(frame, value, origin, *args)

        def normalized(x, y):
            return {"x": x / 999, "y": y / 649}

        main.app.state.video_config = {
            "line": ((100, 50), (300, 50)), "in_transition": (-1, 1)
        }
        main.app.state.stop_line = [normalized(100, 50), normalized(300, 50)]
        main.app.state.gates = [
            {
                "gate": f"Gate {index + 1}", "station_m": index * 10,
                "lane": 1,
                "line": {"L": normalized(100, y), "R": normalized(300, y)},
            }
            for index, y in enumerate((100, 200))
        ]
        main.app.state.road_boundaries = None
        main.app.state.road_marking_result = None
        main.app.state.road_marking_reload_requested = False
        main.app.state.processing_started = True
        main.app.state.cancel_requested = False
        try:
            with patch.object(main, "YOLO", CrossingModel), \
                 patch.object(main, "open_video_capture", return_value=FakeCapture(11)), \
                 patch.object(main, "road_markings_for_frame", return_value=None), \
                 patch.object(main, "estimate_camera_motion", return_value=(None, 0, 0)), \
                 patch.object(main.frame_broadcaster, "publish"), \
                 patch.object(main.frame_broadcaster, "finish"), \
                 patch.object(main.cv2, "putText", side_effect=collect_text):
                main.run_monitor(
                    "cpu", main.MODEL_PATH, None, 1280, 0.38,
                    "test.mp4", "test.mp4", False, "auto", 1000, 650, 10.0,
                )
        finally:
            main.app.state._state.clear()
            main.app.state._state.update(saved_state)
            main.runtime_status.clear()
            main.runtime_status.update(saved_status)

        sections = [(frame, value, color) for frame, value, _, color in text_calls
                    if value.startswith("SECTION ")]
        self.assertEqual({frame for frame, _, _ in sections}, {10, 11})
        self.assertTrue(any(frame >= 10 and "km/h G1-G2" in value and "--" not in value
                            for frame, value, _ in sections))
        self.assertTrue(all(color == (0, 0, 255) for _, _, color in sections))
        live = [(frame, value, color) for frame, value, _, color in text_calls
                if value.startswith("EST ")]
        self.assertEqual({frame for frame, _, _ in live}, {5, 6, 7, 8, 9})
        self.assertTrue(all(color == (0, 255, 255) for _, _, color in live))
        self.assertFalse(any(value.startswith("last dectected: ")
                             for _, value, _, _ in text_calls))

    def test_two_vehicles_without_calibration_show_provisional_speeds(self):
        class UntrackedVehiclesModel:
            def __init__(self, *_args, **_kwargs):
                self.frame_index = 0

            def track(self, *_args, **_kwargs):
                bottom = 300 + 20 * self.frame_index
                self.frame_index += 1
                boxes = type("Boxes", (), {
                    "xyxy": FakeTensor([
                        [180, bottom - 40, 220, bottom],
                        [480, bottom - 40, 520, bottom],
                    ]),
                    "id": None,
                    "cls": FakeTensor([2, 2]),
                    "conf": FakeTensor([0.9, 0.9]),
                })()
                return [type("Result", (), {"boxes": boxes})()]

        saved_state = dict(main.app.state._state)
        saved_status = dict(main.runtime_status)
        text_calls = []
        original_put_text = cv2.putText

        def collect_text(frame, value, origin, *args):
            text_calls.append((value, origin))
            return original_put_text(frame, value, origin, *args)

        def normalized(x, y):
            return {"x": x / 999, "y": y / 649}

        main.app.state.video_config = {
            "line": ((100, 250), (600, 250)), "in_transition": (-1, 1)
        }
        main.app.state.stop_line = [normalized(100, 250), normalized(600, 250)]
        main.app.state.gates = [
            {
                "gate": f"Gate {index + 1}", "station_m": index * 10,
                "lane": 1,
                "line": {"L": normalized(100, y), "R": normalized(600, y)},
            }
            for index, y in enumerate((300, 500))
        ]
        main.app.state.road_boundaries = None
        main.app.state.road_marking_result = None
        main.app.state.road_marking_reload_requested = False
        main.app.state.processing_started = True
        main.app.state.cancel_requested = False
        try:
            with patch.object(main, "YOLO", UntrackedVehiclesModel), \
                 patch.object(main, "open_video_capture", return_value=FakeCapture()), \
                 patch.object(main, "road_markings_for_frame", return_value=None), \
                 patch.object(main, "estimate_camera_motion", return_value=(None, 0, 0)), \
                 patch.object(main.frame_broadcaster, "publish"), \
                 patch.object(main.frame_broadcaster, "finish"), \
                 patch.object(main.cv2, "putText", side_effect=collect_text):
                main.run_monitor(
                    "cpu", main.MODEL_PATH, None, 1280, 0.38,
                    "test.mp4", "test.mp4", False, "auto", 1000, 650, 10.0,
                )
        finally:
            main.app.state._state.clear()
            main.app.state._state.update(saved_state)
            main.runtime_status.clear()
            main.runtime_status.update(saved_status)

        estimated_labels = [
            (value, origin[0]) for value, origin in text_calls
            if value.startswith("EST ") and "(estimate)" in value
        ]
        for box_x in (180, 480):
            self.assertTrue(any(
                x == box_x and "--" not in label and "km/h" in label
                for label, x in estimated_labels
            ))
            self.assertTrue(any(
                origin[0] == box_x and value.startswith("EST ")
                and "--" not in value and "(estimate)" not in value
                for value, origin in text_calls
            ))

    def test_traffic_light_updates_without_tracking_ids(self):
        class OneFrameCapture(FakeCapture):
            def __init__(self):
                super().__init__()
                self.frames = self.frames[:1]

        class UntrackedLightModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def track(self, *_args, **_kwargs):
                boxes = type("Boxes", (), {
                    "xyxy": FakeTensor([[100, 100, 130, 160]]),
                    "id": None,
                    "cls": FakeTensor([9]),
                    "conf": FakeTensor([0.9]),
                })()
                return [type("Result", (), {"boxes": boxes})()]

        class FakeLightDetector:
            COLORS = {"UNKNOWN": (0, 0, 0), "GREEN": (0, 255, 0)}
            updates = 0

            def update(self, roi):
                self.__class__.updates += 1
                return "GREEN", self.COLORS["GREEN"], 0.0, 1.0

        saved_state = dict(main.app.state._state)
        saved_status = dict(main.runtime_status)
        text_calls = []
        original_put_text = cv2.putText

        def collect_text(frame, value, origin, *args):
            text_calls.append(value)
            return original_put_text(frame, value, origin, *args)

        main.app.state.video_config = {
            "line": ((100, 50), (300, 50)), "in_transition": (-1, 1)
        }
        main.app.state.stop_line = [
            {"x": 100 / 999, "y": 50 / 649},
            {"x": 300 / 999, "y": 50 / 649},
        ]
        main.app.state.gates = []
        main.app.state.road_boundaries = None
        main.app.state.road_marking_result = None
        main.app.state.road_marking_reload_requested = False
        main.app.state.processing_started = True
        main.app.state.cancel_requested = False
        try:
            with patch.object(main, "YOLO", UntrackedLightModel), \
                 patch.object(main, "TrafficLightDetector", FakeLightDetector), \
                 patch.object(main, "open_video_capture", return_value=OneFrameCapture()), \
                 patch.object(main, "road_markings_for_frame", return_value=None), \
                 patch.object(main, "estimate_camera_motion", return_value=(None, 0, 0)), \
                 patch.object(main.frame_broadcaster, "publish"), \
                 patch.object(main.frame_broadcaster, "finish"), \
                 patch.object(main.cv2, "putText", side_effect=collect_text):
                main.run_monitor(
                    "cpu", main.MODEL_PATH, None, 1280, 0.38,
                    "test.mp4", "test.mp4", False, "auto", 1000, 650, 10.0,
                )
        finally:
            main.app.state._state.clear()
            main.app.state._state.update(saved_state)
            main.runtime_status.clear()
            main.runtime_status.update(saved_status)

        self.assertEqual(FakeLightDetector.updates, 1)
        self.assertIn("GREEN", text_calls)

    def test_gate_distance_speed_stays_above_vehicle_box_across_gate_pair(self):
        saved_state = dict(main.app.state._state)
        saved_status = dict(main.runtime_status)
        text_calls = []
        original_put_text = cv2.putText

        def collect_text(frame, text, origin, *args):
            text_calls.append((text, origin))
            return original_put_text(frame, text, origin, *args)

        def normalized(x, y):
            return {"x": x / 999, "y": y / 649}

        main.app.state.video_config = {
            "line": ((100, 50), (300, 50)), "in_transition": (-1, 1)
        }
        main.app.state.stop_line = [normalized(100, 50), normalized(300, 50)]
        main.app.state.gates = [
            {
                "gate": f"Gate {index + 1}",
                "station_m": index * 10,
                "lane": 1,
                "line": {
                    "L": normalized(100, y),
                    "R": normalized(300, y),
                },
            }
            for index, y in enumerate((100, 200))
        ]
        main.app.state.road_boundaries = None
        main.app.state.road_marking_result = None
        main.app.state.road_marking_reload_requested = False
        main.app.state.processing_started = True
        main.app.state.cancel_requested = False
        try:
            with patch.object(main, "YOLO", FakeModel), \
                 patch.object(main, "open_video_capture", return_value=FakeCapture()), \
                 patch.object(main, "road_markings_for_frame", return_value=None), \
                 patch.object(main, "estimate_camera_motion", return_value=(None, 0, 0)), \
                 patch.object(main.frame_broadcaster, "publish"), \
                 patch.object(main.frame_broadcaster, "finish"), \
                 patch.object(main.cv2, "putText", side_effect=collect_text):
                main.run_monitor(
                    "cpu", main.MODEL_PATH, None, 1280, 0.38,
                    "test.mp4", "test.mp4", False, "auto", 1000, 650, 10.0,
                )
        finally:
            main.app.state._state.clear()
            main.app.state._state.update(saved_state)
            main.runtime_status.clear()
            main.runtime_status.update(saved_status)

        speed_labels = [(text, origin) for text, origin in text_calls if text.startswith("EST ")]
        self.assertTrue(speed_labels)
        numeric_labels = [(label, origin) for label, origin in speed_labels if "--" not in label]
        self.assertTrue(numeric_labels)
        self.assertTrue(any("(estimate)" in label for label, _ in numeric_labels))
        self.assertTrue(any("(estimate)" not in label for label, _ in numeric_labels))
        self.assertFalse(any(label.startswith("EST AVG") for label, _ in speed_labels))
        self.assertIn("km/h", numeric_labels[0][0])
        self.assertLess(numeric_labels[0][1][1], 180)


if __name__ == "__main__":
    unittest.main()
