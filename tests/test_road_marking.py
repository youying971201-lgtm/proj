import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from road_marking import RoadMarkingResult, draw_road_markings, extract_marking_coordinates


class RoadMarkingCoordinateTests(unittest.TestCase):
    def test_cached_overlay_updates_when_selected_classes_change(self):
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        class_map = np.zeros((12, 12), dtype=np.uint8)
        class_map[2, 2] = 4
        class_map[8, 8] = 9
        result = RoadMarkingResult(class_map, [], 0.0, (12, 12))

        solid_only = draw_road_markings(frame, result, class_ids=(4,))
        dashed_only = draw_road_markings(frame, result, class_ids=(9,))

        self.assertTrue(np.any(solid_only[2, 2] != 0))
        self.assertTrue(np.all(solid_only[8, 8] == 0))
        self.assertTrue(np.any(dashed_only[8, 8] != 0))
        self.assertTrue(np.all(dashed_only[2, 2] == 0))

    def test_dashed_line_keeps_segments_and_builds_full_polyline(self):
        class_map = np.zeros((180, 240), dtype=np.uint8)
        for top, left in ((130, 120), (90, 115), (50, 110)):
            cv2.rectangle(class_map, (left, top), (left + 7, top + 16), 9, -1)

        markings = extract_marking_coordinates(class_map, class_ids=(9,))

        self.assertEqual(len(markings), 1)
        self.assertEqual(len(markings[0]["components"]), 3)
        self.assertEqual(len(markings[0]["polylines"]), 1)
        self.assertGreaterEqual(len(markings[0]["polylines"][0]["points"]), 2)

    def test_area_coordinates_include_pixels_and_normalized_values(self):
        class_map = np.zeros((100, 200), dtype=np.uint8)
        cv2.rectangle(class_map, (20, 40), (80, 60), 2, -1)

        component = extract_marking_coordinates(class_map, class_ids=(2,))[0][
            "components"
        ][0]

        self.assertEqual(component["bbox"]["x"], 20)
        self.assertTrue(0 <= component["centroid"]["nx"] <= 1)
        self.assertTrue(0 <= component["centroid"]["ny"] <= 1)


if __name__ == "__main__":
    unittest.main()
