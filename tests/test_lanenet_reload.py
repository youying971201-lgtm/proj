import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import main
from road_marking import RoadMarkingResult, draw_road_markings


class LaneNetReloadTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = dict(main.app.state._state)
        self.saved_status = dict(main.road_marking_status)
        self.frame = np.zeros((16, 16, 3), dtype=np.uint8)
        class_map = np.zeros((16, 16), dtype=np.uint8)
        class_map[4:12, 7] = 4
        self.cached = RoadMarkingResult(class_map, [], 12.0, (16, 16))
        main.app.state.road_marking_result = self.cached
        main.app.state.road_marking_reload_requested = False
        main.app.state.road_model_path = "stage5.pth"
        main.app.state.processing_started = True
        main.road_marking_status.update(
            enabled=True, state="ready", message="cached", result=self.cached.as_dict(1)
        )

    def tearDown(self):
        main.app.state._state.clear()
        main.app.state._state.update(self.saved_state)
        main.road_marking_status.clear()
        main.road_marking_status.update(self.saved_status)

    @patch("main.reload_road_markings")
    def test_only_one_inference_per_reload_and_cached_result_stays_visible(self, reload_model):
        checkpoint = Path("stage5.pth")
        self.assertIs(
            main.road_markings_for_frame(self.frame, 2, checkpoint, "cpu", 512),
            self.cached,
        )
        reload_model.assert_not_called()

        main.app.state.road_marking_reload_requested = True
        refreshed = RoadMarkingResult(self.cached.class_map.copy(), [], 18.0, (16, 16))
        reload_model.return_value = refreshed
        self.assertIs(
            main.road_markings_for_frame(self.frame, 3, checkpoint, "cpu", 512),
            refreshed,
        )
        self.assertIs(
            main.road_markings_for_frame(self.frame, 4, checkpoint, "cpu", 512),
            refreshed,
        )
        reload_model.assert_called_once_with(self.frame, checkpoint, "cpu", 512)
        self.assertEqual(main.road_marking_status["result"]["frame_index"], 3)

        first_overlay = draw_road_markings(self.frame, refreshed)
        second_overlay = draw_road_markings(self.frame, refreshed)
        np.testing.assert_array_equal(first_overlay, second_overlay)
        self.assertTrue(np.any(first_overlay != self.frame))

    @patch("main.reload_road_markings", side_effect=RuntimeError("bad checkpoint"))
    def test_failed_reload_keeps_previous_markings(self, reload_model):
        main.app.state.road_marking_reload_requested = True
        result = main.road_markings_for_frame(
            self.frame, 3, Path("stage5.pth"), "cpu", 512
        )
        self.assertIs(result, self.cached)
        self.assertIs(main.app.state.road_marking_result, self.cached)
        self.assertEqual(main.road_marking_status["state"], "error")
        reload_model.assert_called_once()

    def test_reload_endpoint_requires_running_model_and_rejects_duplicate(self):
        main.app.state.processing_started = False
        with self.assertRaises(HTTPException) as stopped:
            main.request_lanenet_reload()
        self.assertEqual(stopped.exception.status_code, 409)

        main.app.state.processing_started = True
        self.assertEqual(main.request_lanenet_reload()["status"], "queued")
        self.assertTrue(main.app.state.road_marking_reload_requested)
        with self.assertRaises(HTTPException) as duplicate:
            main.request_lanenet_reload()
        self.assertEqual(duplicate.exception.status_code, 409)

    def test_stage5_uses_cpu_for_mps_and_cuda_for_cuda(self):
        self.assertEqual(main.resolve_road_marking_device("mps"), "cpu")
        self.assertEqual(main.resolve_road_marking_device("cuda"), "cuda")
        self.assertEqual(main.resolve_road_marking_device("cpu"), "cpu")

    def test_yolo_device_falls_back_to_cpu_when_requested_accelerator_is_unavailable(self):
        with patch.object(main.torch.cuda, "is_available", return_value=False), \
             patch.object(main.torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(main.resolve_device("auto"), "cpu")
            self.assertEqual(main.resolve_device("cuda"), "cpu")
            self.assertEqual(main.resolve_device("mps"), "cpu")

        with patch.object(main.torch.cuda, "is_available", return_value=True), \
             patch.object(main.torch.backends.mps, "is_available", return_value=True):
            self.assertEqual(main.resolve_device("auto"), "cuda")
            self.assertEqual(main.resolve_road_marking_device("cuda"), "cuda")


if __name__ == "__main__":
    unittest.main()
