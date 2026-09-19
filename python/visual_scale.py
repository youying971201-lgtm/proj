"""從多車的三維尺寸先驗估計道路平面尺度；結果僅供預估。"""

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite, log

import numpy as np
from scipy.optimize import minimize


# COCO 車種的粗略長、寬、高（公尺）。這些是統計先驗，不是個別車輛的實測尺寸。
VEHICLE_PRIORS = {
    2: (4.4, 1.8, 1.5),   # car
    3: (2.1, 0.8, 1.55),  # motorcycle + rider；變異較大
    5: (10.0, 2.5, 3.2),  # bus
    7: (7.0, 2.4, 3.0),   # truck
}


@dataclass(frozen=True)
class BoxObservation:
    track_id: int
    pair_index: int
    box: tuple[float, float, float, float]
    class_id: int
    vehicle_size: tuple[float, float, float]


class VisualScaleCalibrator:
    """共同擬合相機焦距、道路尺度與多車近似三維框。"""

    def __init__(
        self, sections, frame_size: tuple[int, int], lane_count: int,
        expected_pairs: int | None = None,
    ):
        self.width, self.height = frame_size
        self.lane_count = max(lane_count, 1)
        self.sections = {
            section.gate_pair_index: section
            for section in sections
            if section.first.name != "Stop line" and section.second.name != "Stop line"
        }
        if expected_pairs is not None and sorted(self.sections) != list(range(expected_pairs)):
            self.sections.clear()
        self.image_matrices = {}
        for index, section in self.sections.items():
            matrix = np.linalg.inv(section.transform)
            midpoint = np.array([
                0.5, (section.first.station_m + section.second.station_m) / 2, 1,
            ])
            self.image_matrices[index] = matrix if (matrix @ midpoint)[2] > 0 else -matrix
        self.observations: deque[BoxObservation] = deque(maxlen=32)
        self.last_observed_frame: dict[int, int] = {}
        self.last_fit_frame = -1000
        self.road_width_m: float | None = None
        self.focal_px: float | None = None
        self.relative_uncertainty: float | None = None
        self.section_scales: dict[int, float] = {}
        # 沒有足夠車輛時的暫估值；只使用車道寬與視角先驗，不借用預填的 10 m。
        self.prior_road_width_m = min(24.0, max(3.5, self.lane_count * 3.5))
        self.prior_focal_px = 1.3 * self.width
        self.provisional_scales = {}
        for index in self.sections:
            camera = self._camera(index, self.prior_focal_px, self.prior_road_width_m)
            if camera is not None:
                self.provisional_scales[index] = camera[2]
        self.revision = 0

    @property
    def ready(self) -> bool:
        return self.road_width_m is not None

    @property
    def provisional_available(self) -> bool:
        return bool(self.sections) and len(self.provisional_scales) == len(self.sections)

    @property
    def confidence(self) -> str:
        if not self.ready or self.relative_uncertainty is None:
            return "CAL"
        distinct_vehicles = len({item.track_id for item in self.observations})
        reliable_vehicles = len({
            item.track_id for item in self.observations if item.class_id != 3
        })
        return (
            "MED" if distinct_vehicles >= 8 and reliable_vehicles >= 5
            and self.relative_uncertainty <= 0.25 else "LOW"
        )

    def as_dict(self) -> dict:
        return {
            "state": "ready" if self.ready else "calibrating",
            "method": "vehicle_size_prior",
            "vehicles": len({item.track_id for item in self.observations}),
            "quality": self.confidence,
            "provisional_available": self.provisional_available,
            "relative_fit_dispersion": self.relative_uncertainty,
            "estimated_road_width_m": self.road_width_m,
            "estimated_focal_px": self.focal_px,
        }

    def _camera(self, pair_index: int, focal_px: float, road_width_m: float):
        matrix = self.image_matrices[pair_index]
        cx, cy = (self.width - 1) / 2, (self.height - 1) / 2
        inverse_intrinsics = np.array([
            [1 / focal_px, 0, -cx / focal_px],
            [0, 1 / focal_px, -cy / focal_px],
            [0, 0, 1],
        ])
        lateral = inverse_intrinsics @ matrix[:, 0]
        longitudinal = inverse_intrinsics @ matrix[:, 1]
        lateral_norm = np.linalg.norm(lateral)
        longitudinal_norm = np.linalg.norm(longitudinal)
        normal = np.cross(lateral, longitudinal)
        normal_norm = np.linalg.norm(normal)
        if min(lateral_norm, longitudinal_norm, normal_norm) < 1e-8:
            return None
        axis_cosine = float(np.dot(lateral, longitudinal) / (
            lateral_norm * longitudinal_norm
        ))
        if abs(axis_cosine) >= 0.96:
            return None
        normal /= normal_norm
        intrinsics = np.array([
            [focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1],
        ])
        vertical = lateral_norm / road_width_m * (intrinsics @ normal)
        section = self.sections[pair_index]
        midpoint = np.array([0.5, (section.first.station_m + section.second.station_m) / 2, 1])
        ground = matrix @ midpoint
        top = ground + vertical
        if abs(top[2]) < 1e-9:
            return None
        if top[1] / top[2] > ground[1] / ground[2]:
            vertical = -vertical
        meter_per_station = road_width_m * longitudinal_norm / lateral_norm
        return matrix, vertical, meter_per_station, axis_cosine

    @staticmethod
    def _project_box(camera, center, size, road_width_m: float):
        matrix, vertical, meter_per_station, axis_cosine = camera
        length, width, height = size
        if meter_per_station <= 0:
            return None
        axis_sine = np.sqrt(1 - axis_cosine ** 2)
        points = []
        for across_m in (-width / 2, width / 2):
            for along_m in (-length / 2, length / 2):
                # Gate 與車道方向可斜交；車身長寬仍取道路平面上的正交軸。
                across = (across_m - along_m * axis_cosine / axis_sine) / road_width_m
                along = along_m / (meter_per_station * axis_sine)
                ground = matrix @ np.array([center[0] + across, center[1] + along, 1.0])
                for elevation in (0, height):
                    projected = ground + elevation * vertical
                    if projected[2] <= 1e-9:
                        return None
                    points.append(projected[:2] / projected[2])
        coordinates = np.asarray(points)
        if not np.isfinite(coordinates).all():
            return None
        return (
            float(coordinates[:, 0].min()), float(coordinates[:, 1].min()),
            float(coordinates[:, 0].max()), float(coordinates[:, 1].max()),
        )

    def _box_error(self, observation: BoxObservation, camera, road_width_m: float) -> float:
        x1, y1, x2, y2 = observation.box
        observed_width, observed_height = x2 - x1, y2 - y1
        section = self.sections[observation.pair_index]
        ground = section.transform @ np.array([(x1 + x2) / 2, y2, 1.0])
        if abs(ground[2]) < 1e-9:
            return 10.0
        anchor = ground[:2] / ground[2]
        length, width, _ = observation.vehicle_size
        axis_cosine = camera[3]
        axis_sine = np.sqrt(1 - axis_cosine ** 2)
        best = 10.0
        for along_m in (-length / 2, 0.0, length / 2):
            for across_m in (-width / 4, 0.0, width / 4):
                across = (across_m - along_m * axis_cosine / axis_sine) / road_width_m
                along = along_m / (camera[2] * axis_sine)
                predicted = self._project_box(
                    camera, (anchor[0] + across, anchor[1] + along),
                    observation.vehicle_size, road_width_m,
                )
                if predicted is None:
                    continue
                px1, py1, px2, py2 = predicted
                predicted_width, predicted_height = px2 - px1, py2 - py1
                if predicted_width <= 0 or predicted_height <= 0:
                    continue
                error = (
                    log(predicted_width / observed_width) ** 2
                    + log(predicted_height / observed_height) ** 2
                    + 0.35 * ((py2 - y2) / observed_height) ** 2
                    + 0.15 * (((px1 + px2 - x1 - x2) / 2) / observed_width) ** 2
                )
                best = min(best, error)
        return best

    def _fit_score(self, focal_px: float, road_width_m: float):
        cameras = {
            index: self._camera(index, focal_px, road_width_m)
            for index in self.sections
        }
        if any(camera is None for camera in cameras.values()):
            return float("inf"), []
        reliable = [item for item in self.observations if item.class_id != 3]
        selected = (
            reliable if len({item.track_id for item in reliable}) >= 3 and len(reliable) >= 8
            else self.observations
        )
        by_track = defaultdict(list)
        for observation in selected:
            camera = cameras[observation.pair_index]
            by_track[observation.track_id].append(
                self._box_error(observation, camera, road_width_m)
            )
        if len(by_track) < 3:
            return float("inf"), []
        track_errors = [float(np.median(values)) for values in by_track.values()]
        # 中位數限制單一車型或長軌跡主導整體尺度。
        score = float(np.median(track_errors))
        score += 0.015 * log(focal_px / (1.3 * self.width)) ** 2
        return score, track_errors

    def _fit(self) -> None:
        width_min = max(2.2, min(18.0, self.lane_count * 2.0))
        width_max = min(24.0, max(9.0, width_min + 1.0, self.lane_count * 5.5))
        focal_candidates = self.width * np.array([0.8, 1.1, 1.5, 2.0, 2.8])
        width_candidates = np.linspace(width_min, width_max, 9)
        best = None
        for focal in focal_candidates:
            for road_width in width_candidates:
                score, errors = self._fit_score(float(focal), float(road_width))
                if best is None or score < best[0]:
                    best = score, float(focal), float(road_width), errors
        if best is None or not isfinite(best[0]):
            return
        refined = minimize(
            lambda values: self._fit_score(float(values[0]), float(values[1]))[0],
            x0=(best[1], best[2]),
            method="Nelder-Mead",
            bounds=((0.65 * self.width, 3.2 * self.width), (width_min, width_max)),
            options={"maxiter": 55, "xatol": 2.0, "fatol": 1e-5},
        )
        score, errors = self._fit_score(float(refined.x[0]), float(refined.x[1]))
        if (
            not isfinite(score)
            or not all(isfinite(float(value)) for value in refined.x)
            or score > 0.45
        ):
            return
        focal, road_width = map(float, refined.x)
        spread = float(np.median(np.sqrt(np.asarray(errors))))
        uncertainty = max(0.2, min(0.75, 0.2 + spread))
        if self.ready:
            change = abs(road_width - self.road_width_m) / self.road_width_m
            uncertainty = max(uncertainty, min(0.75, change))
            focal = 0.8 * self.focal_px + 0.2 * focal
            road_width = 0.8 * self.road_width_m + 0.2 * road_width
        section_scales = {}
        for index in self.sections:
            camera = self._camera(index, focal, road_width)
            if camera is None:
                return
            section_scales[index] = camera[2]
        self.focal_px = focal
        self.road_width_m = road_width
        self.section_scales = section_scales
        # 車體尺寸先驗本身有變異；即使框擬合得很好也不可宣稱高可信度。
        self.relative_uncertainty = uncertainty
        self.revision += 1

    def observe(
        self, track_id: int, box, class_id: int, confidence: float, frame_number: int
    ) -> None:
        if track_id < 0 or class_id not in VEHICLE_PRIORS or confidence < 0.65:
            return
        x1, y1, x2, y2 = (float(value) for value in box)
        if (
            min(x1, y1) < 3 or x2 >= self.width - 3 or y2 >= self.height - 3
            or x2 - x1 < 22 or y2 - y1 < 22
            or frame_number - self.last_observed_frame.get(track_id, -1000) < 12
        ):
            return
        bottom = np.array([(x1 + x2) / 2, y2, 1.0])
        candidate = None
        for index, section in self.sections.items():
            projected = section.transform @ bottom
            if abs(projected[2]) < 1e-9:
                continue
            across, station = projected[:2] / projected[2]
            fraction = (station - section.first.station_m) / (
                section.second.station_m - section.first.station_m
            )
            if 0.05 <= across <= 0.95 and 0.05 <= fraction <= 0.95:
                candidate = index
                break
        if candidate is None:
            return
        self.observations.append(BoxObservation(
            track_id, candidate, (x1, y1, x2, y2),
            class_id, VEHICLE_PRIORS[class_id],
        ))
        self.last_observed_frame[track_id] = frame_number
        reliable = [item for item in self.observations if item.class_id != 3]
        reliable_ready = (
            len({item.track_id for item in reliable}) >= 3 and len(reliable) >= 8
        )
        motorcycle_ready = (
            len({item.track_id for item in self.observations}) >= 5
            and len(self.observations) >= 12
        )
        if (
            frame_number - self.last_fit_frame >= (180 if self.ready else 45)
            and (reliable_ready or motorcycle_ready)
        ):
            self.last_fit_frame = frame_number
            self._fit()

    def _meter_per_station(self, pair_index: int) -> float | None:
        if pair_index not in self.sections:
            return None
        if not self.ready:
            return self.provisional_scales.get(pair_index) if self.provisional_available else None
        if pair_index in self.section_scales:
            return self.section_scales[pair_index]
        camera = self._camera(pair_index, self.focal_px, self.road_width_m)
        return camera[2] if camera is not None else None

    def metric_station(self, station: float) -> float | None:
        if not self.ready and not self.provisional_available:
            return None
        if not self.sections:
            return None
        sections = [self.sections[index] for index in sorted(self.sections)]
        total = 0.0
        for section in sections:
            scale = self._meter_per_station(section.gate_pair_index)
            if scale is None:
                return None
            if station < section.first.station_m:
                return total + (station - section.first.station_m) * scale
            if station <= section.second.station_m:
                return total + (station - section.first.station_m) * scale
            total += (section.second.station_m - section.first.station_m) * scale
        final_scale = self._meter_per_station(sections[-1].gate_pair_index)
        return total + (station - sections[-1].second.station_m) * final_scale

    def lane_ratio(self, pair_index: int, left_path, right_path, across: float) -> float | None:
        if pair_index not in self.sections:
            return None
        section = self.sections[pair_index]
        road_width = self.road_width_m if self.ready else self.prior_road_width_m
        focal = self.focal_px if self.ready else self.prior_focal_px
        camera = self._camera(pair_index, focal, road_width)
        if camera is None:
            return None
        meter_per_station = camera[2]
        axis_cosine = camera[3]

        def length_at(fraction):
            points = []
            for left, right in zip(left_path, right_path):
                image = np.array([
                    left[0] * (1 - fraction) + right[0] * fraction,
                    left[1] * (1 - fraction) + right[1] * fraction, 1.0,
                ])
                projected = section.transform @ image
                if abs(projected[2]) < 1e-9:
                    return None
                u, station = projected[:2] / projected[2]
                points.append((u * road_width, station * meter_per_station))
            return sum(
                np.sqrt(max(
                    0.0,
                    (second[0] - first[0]) ** 2
                    + (second[1] - first[1]) ** 2
                    + 2 * axis_cosine
                    * (second[0] - first[0]) * (second[1] - first[1]),
                ))
                for first, second in zip(points, points[1:])
            )

        center_length = length_at(0.5)
        lane_length = length_at(across)
        if center_length is None or lane_length is None or center_length < 0.5:
            return None
        ratio = lane_length / center_length
        return ratio if isfinite(ratio) and 0.75 <= ratio <= 1.25 else None
