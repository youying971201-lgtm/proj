import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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


class VehicleModelTests(unittest.TestCase):
    def test_detection_modes_request_expected_classes(self):
        self.assertEqual(main.LEAVING_PRIMARY_CLASSES, [0, 9])
        self.assertEqual(main.ENTERING_PRIMARY_CLASSES, [0, 2, 5, 7, 9])
        self.assertEqual(main.ENTERING_PRIMARY_VEHICLE_CLASSES, {2, 5, 7})
        self.assertEqual(main.ENTERING_CUSTOM_VEHICLE_CLASSES, {4})
        self.assertEqual(main.CUSTOM_VEHICLE_CLASSES, {2, 4})
        self.assertEqual(
            main.detection_plan("entering"),
            (
                main.ENTERING_PRIMARY_CLASSES,
                main.ENTERING_PRIMARY_VEHICLE_CLASSES,
                main.ENTERING_CUSTOM_VEHICLE_CLASSES,
            ),
        )
        self.assertEqual(
            main.detection_plan("leaving"),
            (main.LEAVING_PRIMARY_CLASSES, set(), main.CUSTOM_VEHICLE_CLASSES),
        )
        with self.assertRaises(ValueError):
            main.detection_plan("sideways")

    def test_custom_model_keeps_only_car_and_scooter(self):
        boxes = type("Boxes", (), {
            "xyxy": FakeTensor([
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                [130, 140, 150, 160],
            ]),
            "id": FakeTensor([11, 12, 13, 14]),
            "cls": FakeTensor([2, 4, 5, 3]),
        })()
        result = type("Result", (), {"boxes": boxes})()

        detections = main.tracked_vehicle_detections(
            result, main.CUSTOM_VEHICLE_CLASSES
        )

        self.assertEqual([track_id for _, track_id, _ in detections], [11, 12])
        np.testing.assert_array_equal(detections[0][0], [10, 20, 30, 40])
        np.testing.assert_array_equal(detections[1][0], [50, 60, 70, 80])

    def test_custom_model_without_bytetrack_id_stays_temporary(self):
        boxes = type("Boxes", (), {
            "xyxy": FakeTensor([[10, 20, 30, 40]]),
            "id": None,
            "cls": FakeTensor([2]),
        })()
        result = type("Result", (), {"boxes": boxes})()

        detections = main.tracked_vehicle_detections(
            result, main.CUSTOM_VEHICLE_CLASSES
        )

        self.assertEqual(detections[0][1], -1)

    def test_model_track_ids_use_separate_namespaces(self):
        boxes = type("Boxes", (), {
            "xyxy": FakeTensor([[10, 20, 30, 40]]),
            "id": FakeTensor([11]),
            "cls": FakeTensor([2]),
        })()
        result = type("Result", (), {"boxes": boxes})()

        primary = main.tracked_vehicle_detections(result, {2}, track_namespace=0)
        custom = main.tracked_vehicle_detections(result, {2}, track_namespace=1)

        self.assertEqual(primary[0][1], 22)
        self.assertEqual(custom[0][1], 23)

    def test_custom_scooter_suppresses_overlapping_primary_vehicle(self):
        primary = [
            (np.asarray([10, 10, 50, 50]), 2, None),
            (np.asarray([70, 10, 110, 50]), 4, None),
        ]
        custom = [(np.asarray([12, 12, 48, 50]), 3, None)]

        merged = main.merge_vehicle_detections(primary, custom)

        self.assertEqual([track_id for _, track_id, _ in merged], [4, 3])

    def test_pedestrian_is_drawn_without_entering_vehicle_pipeline(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.asarray([[10, 20, 30, 60], [40, 20, 70, 70]])
        classes = np.asarray([0, 2])
        confidences = np.asarray([0.876, 0.95])

        with patch.object(main.cv2, "rectangle") as rectangle, \
             patch.object(main.cv2, "putText") as put_text:
            main.draw_pedestrians(frame, boxes, classes, confidences)

        rectangle.assert_called_once()
        self.assertEqual(rectangle.call_args.args[1:3], ((10, 20), (30, 60)))
        put_text.assert_called_once()
        self.assertEqual(put_text.call_args.args[1], "PERSON 0.88")


if __name__ == "__main__":
    unittest.main()
