import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from video_source import open_video_capture


class VideoSourceTests(unittest.TestCase):
    @patch("video_source.cv2.VideoCapture")
    def test_camera_index_uses_selected_backend_and_settings(self, video_capture):
        cap = Mock()
        video_capture.return_value = cap

        result = open_video_capture(
            1,
            camera_backend="dshow",
            camera_width=1920,
            camera_height=1080,
            camera_fps=30,
        )

        video_capture.assert_called_once_with(1, cv2.CAP_DSHOW)
        self.assertIs(result, cap)
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
        cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)

    @patch("video_source.cv2.VideoCapture")
    def test_file_source_remains_a_path(self, video_capture):
        source = r"C:\video\sample.mp4"

        open_video_capture(source, camera_backend="dshow")

        video_capture.assert_called_once_with(source)


if __name__ == "__main__":
    unittest.main()
