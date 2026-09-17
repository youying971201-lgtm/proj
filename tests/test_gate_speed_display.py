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
    def __init__(self):
        self.position = 0
        self.frames = [np.zeros((650, 1000, 3), dtype=np.uint8) for _ in range(6)]

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
        bottom = [80, 120, 180, 220, 240, 260][self.frame_index]
        self.frame_index += 1
        boxes = type("Boxes", (), {
            "xyxy": FakeTensor([[180, bottom - 40, 220, bottom]]),
            "id": FakeTensor([42]),
            "cls": FakeTensor([2]),
            "conf": FakeTensor([0.9]),
        })()
        return [type("Result", (), {"boxes": boxes})()]


class GateSpeedDisplayTests(unittest.TestCase):
    def test_completed_gate_pair_draws_estimate_above_vehicle_box(self):
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
        self.assertIn("km/h", speed_labels[0][0])
        self.assertLess(speed_labels[0][1][1], 180)


if __name__ == "__main__":
    unittest.main()
