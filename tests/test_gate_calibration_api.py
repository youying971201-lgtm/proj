import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Request


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import main


def request_for(payload):
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/save_points", "headers": []}, receive)


class GateCalibrationApiTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = dict(main.app.state._state)
        main.app.state.frame_size = (1000, 650)
        main.app.state.video_config = {"line": ((0, 0), (1, 0))}
        main.app.state.source_name = "test.mp4"
        self.temporary = tempfile.TemporaryDirectory()
        self.points_file = Path(self.temporary.name) / "points_data.json"
        self.file_patch = patch.object(main, "POINTS_FILE", self.points_file)
        self.file_patch.start()

    def tearDown(self):
        self.file_patch.stop()
        self.temporary.cleanup()
        main.app.state._state.clear()
        main.app.state._state.update(self.saved_state)

    def payload(self):
        stop = {"L": {"x": 0.1, "y": 0.8}, "R": {"x": 0.9, "y": 0.8}}
        gates = []
        for index in range(5):
            y = 0.7 - index * 0.1
            gates.append({
                "gate": f"Gate {index + 1}",
                "lane": 2,
                "station_m": index * 10,
                "anchor": {"x": 0.5, "y": y},
                "angle_deg": 0,
                "line": {
                    "L": {"x": 0.1 + (0.8 - y) * 0.2, "y": y},
                    "R": {"x": 0.9 - (0.8 - y) * 0.2, "y": y},
                },
            })
        return {
            "schema_version": 3,
            "lane_count": 2,
            "detection_mode": "leaving",
            "stop_line": {"line": stop},
            "road_boundaries": {
                "L": [stop["L"], {"x": 0.24, "y": 0.1}],
                "R": [stop["R"], {"x": 0.76, "y": 0.1}],
            },
            "gates": gates,
        }

    def test_saves_editable_geometry_and_normalized_stop_line(self):
        result = asyncio.run(main.save_points(request_for(self.payload())))
        saved = json.loads(self.points_file.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(saved["schema_version"], 3)
        self.assertEqual(saved["lane_count"], 2)
        self.assertEqual(saved["detection_mode"], "leaving")
        self.assertEqual(main.app.state.detection_mode, "leaving")
        self.assertEqual(saved["video"]["width"], 1000)
        self.assertEqual(saved["gates"][0]["station_m"], 0)
        self.assertEqual(saved["gates"][0]["anchor"], {"x": 0.5, "y": 0.7})
        self.assertEqual(len(saved["road_boundaries"]["L"]), 2)
        self.assertEqual(main.load_saved_line(((0, 0), (0, 0)), (1000, 650)),
                         ((100, 519), (899, 519)))

    def test_rejects_zero_length_gate(self):
        payload = self.payload()
        payload["gates"][0]["line"]["R"] = payload["gates"][0]["line"]["L"]

        with self.assertRaises(HTTPException) as error:
            asyncio.run(main.save_points(request_for(payload)))
        self.assertEqual(error.exception.status_code, 422)
        self.assertFalse(self.points_file.exists())

    def test_normalizes_gate_distance_to_ten_meters(self):
        payload = self.payload()
        for index, gate in enumerate(payload["gates"]):
            gate["station_m"] = index * 25

        asyncio.run(main.save_points(request_for(payload)))
        saved = json.loads(self.points_file.read_text(encoding="utf-8"))
        self.assertEqual(
            [gate["station_m"] for gate in saved["gates"]],
            [0, 10, 20, 30, 40],
        )

    def test_rejects_missing_gate_number_for_ten_meter_scale(self):
        payload = self.payload()
        payload["gates"][1]["gate"] = "Gate 3"

        with self.assertRaises(HTTPException) as error:
            asyncio.run(main.save_points(request_for(payload)))
        self.assertEqual(error.exception.status_code, 422)
        self.assertFalse(self.points_file.exists())

    def test_saves_entering_detection_mode(self):
        payload = self.payload()
        payload["detection_mode"] = "entering"

        asyncio.run(main.save_points(request_for(payload)))
        saved = json.loads(self.points_file.read_text(encoding="utf-8"))

        self.assertEqual(saved["detection_mode"], "entering")
        self.assertEqual(main.app.state.detection_mode, "entering")

    def test_rejects_unknown_detection_mode(self):
        payload = self.payload()
        payload["detection_mode"] = "sideways"

        with self.assertRaises(HTTPException) as error:
            asyncio.run(main.save_points(request_for(payload)))

        self.assertEqual(error.exception.status_code, 422)
        self.assertFalse(self.points_file.exists())


if __name__ == "__main__":
    unittest.main()
