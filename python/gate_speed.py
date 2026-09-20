"""用 Gate 校正的道路位置估算即時速度與跨 Gate 區間均速。"""

from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite
import re

import cv2
import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class Gate:
    name: str
    station_m: float
    left: Point
    right: Point


@dataclass(frozen=True)
class SpeedEstimate:
    kmh: float
    distance_m: float
    duration_s: float
    lane_number: int
    from_gate: str
    to_gate: str
    lane_adjusted: bool
    method: str = "gate"
    confidence: str = "NOMINAL"


@dataclass(frozen=True)
class StationSection:
    first: Gate
    second: Gate
    transform: np.ndarray
    gate_pair_index: int


class KnownGateMetric:
    """以操作員指定的一般 Gate 10 公尺週期換算道路座標。"""

    ready = True
    confidence = "GATE"

    def __init__(
        self, sections: list[StationSection], lane_count: int,
        pair_paths: list[tuple[list[Point], list[Point]]],
    ) -> None:
        self.sections = {
            section.gate_pair_index: section
            for section in sections
            if section.first.name != "Stop line" and section.second.name != "Stop line"
        }
        self.road_width_m = max(lane_count, 1) * 3.5
        self.paths = {
            index: (np.asarray(left, dtype=float), np.asarray(right, dtype=float))
            for index, (left, right) in enumerate(pair_paths)
            if index in self.sections and len(left) > 2 and len(right) > 2
        }
        self.center_lengths = {
            index: self._path_length(index, 0.5) for index in self.paths
        }

    @staticmethod
    def metric_station(station: float) -> float:
        return station

    def station_uncertainty(self, pair_index: int, point: Point, pixels: float = 2.0) -> float:
        """估計接地點移動數個像素時，該段透視映射會改變多少公尺。"""
        section = self.sections.get(pair_index)
        if section is None:
            return 0.0
        matrix = section.transform
        x, y = point
        numerator = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        if abs(denominator) < 1e-9:
            return float("inf")
        center = numerator / denominator
        changes = []
        for offset, axis in ((pixels, 0), (-pixels, 0), (pixels, 1), (-pixels, 1)):
            nearby_denominator = denominator + offset * matrix[2, axis]
            if abs(nearby_denominator) < 1e-9:
                return float("inf")
            nearby = (numerator + offset * matrix[1, axis]) / nearby_denominator
            if not isfinite(nearby):
                return float("inf")
            changes.append(abs(nearby - center))
        return max(changes) if isfinite(center) else float("inf")

    def _path_length(self, pair_index: int, across: float) -> float | None:
        section = self.sections.get(pair_index)
        path = self.paths.get(pair_index)
        if section is None or path is None:
            return None
        left, right = path
        pixels = left * (1 - across) + right * across
        projected = pixels @ section.transform[:, :2].T + section.transform[:, 2]
        denominator = projected[:, 2]
        if np.any(np.abs(denominator) < 1e-9):
            return None
        world = projected[:, :2] / denominator[:, None]
        if not np.isfinite(world).all():
            return None
        world[:, 0] *= self.road_width_m
        return float(np.hypot(np.diff(world[:, 0]), np.diff(world[:, 1])).sum())

    def lane_ratio(self, pair_index: int, left_path, right_path, across: float) -> float | None:
        center_length = self.center_lengths.get(pair_index)
        lane_length = self._path_length(pair_index, across)
        if center_length is None or lane_length is None or center_length < 0.5:
            return None
        ratio = lane_length / center_length
        return ratio if isfinite(ratio) and 0.75 <= ratio <= 1.25 else None

    @staticmethod
    def as_dict() -> dict:
        return {
            "state": "ready", "method": "gate_distance",
            "gate_spacing_m": 10.0, "optional_gate_offset_m": 4.0,
            "quality": "ASSUMED", "vehicles": 0,
        }


@dataclass
class TrackState:
    last_point: Point | None = None
    last_time: float | None = None
    last_frame: int | None = None
    last_crossing: tuple[int, float, float] | None = None
    crossed_gates: set[int] = field(default_factory=set)
    latest: SpeedEstimate | None = None
    section_total: SpeedEstimate | None = None
    section_start: tuple[int, float, float] | None = None
    section_distance_m: float = 0.0
    section_lane_adjusted: bool = True
    local_samples: deque[tuple[float, float, float]] = field(default_factory=deque)
    local_latest: SpeedEstimate | None = None
    rejected_points: int = 0
    filtered_velocity_mps: float | None = None
    velocity_initialized: bool = False
    speed_zone_state: str = "before"
    gate_reference_velocity_mps: float | None = None
    gate_reference_time: float | None = None


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _subtract(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def segment_intersection(
    start: Point, end: Point, left: Point, right: Point
) -> tuple[float, float] | None:
    """回傳車輛線段與有限 Gate 相交時的時間比例、橫向比例。"""
    movement = _subtract(end, start)
    gate = _subtract(right, left)
    denominator = _cross(movement, gate)
    if abs(denominator) < 1e-9:
        return None
    offset = _subtract(left, start)
    time_fraction = _cross(offset, gate) / denominator
    across_fraction = _cross(offset, movement) / denominator
    if not (0 < time_fraction <= 1 and 0 <= across_fraction <= 1):
        return None
    return time_fraction, across_fraction


def _pixel_point(value: dict, width: int, height: int) -> Point:
    return float(value["x"]) * (width - 1), float(value["y"]) * (height - 1)


def known_gate_stations(saved_gates: list[dict]) -> list[float]:
    """一般相鄰 Gate 為 10 m；可選的 X.4 把該段分成 4 m、6 m。"""
    stations = []
    next_regular = 1
    previous_optional = False
    for item in saved_gates:
        match = re.fullmatch(r"Gate (\d+)(\.4)?", str(item["gate"]))
        if match is None:
            raise ValueError("Gate 名稱須為 Gate 1、Gate 1.4、Gate 2 等格式")
        number = int(match.group(1))
        optional = match.group(2) is not None
        if optional:
            if number != next_regular - 1 or previous_optional or not stations:
                raise ValueError("Gate X.4 必須緊接在 Gate X 之後")
            stations.append((number - 1) * 10.0 + 4.0)
        else:
            if number != next_regular:
                raise ValueError("一般 Gate 必須從 Gate 1 起依序排列")
            stations.append((number - 1) * 10.0)
            next_regular += 1
        previous_optional = optional
    return stations


def _point_on_path(path: list[Point], distance: float) -> Point:
    remaining = distance
    for start, end in zip(path, path[1:]):
        length = hypot(end[0] - start[0], end[1] - start[1])
        if length <= 1e-9:
            continue
        if remaining <= length:
            fraction = max(0.0, remaining / length)
            return (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
        remaining -= length
    return path[-1]


def _position_on_path(path: list[Point], point: Point) -> float | None:
    """找到 Gate 端點在道路邊界上的累積像素弧長。"""
    cumulative = 0.0
    best = None
    for start, end in zip(path, path[1:]):
        delta = _subtract(end, start)
        length_squared = delta[0] ** 2 + delta[1] ** 2
        if length_squared <= 1e-9:
            continue
        fraction = max(0.0, min(1.0, (
            (point[0] - start[0]) * delta[0]
            + (point[1] - start[1]) * delta[1]
        ) / length_squared))
        projection = (
            start[0] + fraction * delta[0],
            start[1] + fraction * delta[1],
        )
        error = hypot(point[0] - projection[0], point[1] - projection[1])
        candidate = cumulative + fraction * hypot(*delta)
        if best is None or error < best[0]:
            best = error, candidate
        cumulative += hypot(*delta)
    return best[1] if best is not None and best[0] <= 4.0 else None


class GateSpeedEstimator:
    """每個追蹤 ID 各自記錄 Gate 穿越時間與最近一次區間均速。"""

    def __init__(
        self,
        saved_gates: list[dict],
        frame_size: tuple[int, int],
        road_boundaries: dict | None = None,
        max_track_gap_frames: int = 5,
        max_track_gap_seconds: float = 2.0,
        stop_line: tuple[Point, Point] | None = None,
        metric_calibrator=None,
        known_gate_spacing: bool = False,
    ) -> None:
        width, height = frame_size
        if known_gate_spacing and metric_calibrator is not None:
            raise ValueError("已指定 Gate 距離時不可再套用另一種公尺尺度")
        gate_stations = (
            known_gate_stations(saved_gates) if known_gate_spacing
            else [float(item["station_m"]) for item in saved_gates]
        )
        self.gates = [
            Gate(
                name=str(item["gate"]),
                station_m=station,
                left=_pixel_point(item["line"]["L"], width, height),
                right=_pixel_point(item["line"]["R"], width, height),
            )
            for item, station in zip(saved_gates, gate_stations)
        ]
        self.gate_bounds = [
            (
                min(gate.left[0], gate.right[0]),
                max(gate.left[0], gate.right[0]),
                min(gate.left[1], gate.right[1]),
                max(gate.left[1], gate.right[1]),
            )
            for gate in self.gates
        ]
        self.lane_count = int(saved_gates[0]["lane"]) if saved_gates else 1
        self.max_track_gap_frames = max_track_gap_frames
        self.max_track_gap_seconds = max_track_gap_seconds
        self.metric_calibrator = metric_calibrator
        self.stop_line = stop_line
        self.speed_zone = (
            (stop_line, (self.gates[-1].left, self.gates[-1].right))
            if stop_line is not None and self.gates else None
        )
        self.tracks: dict[int, TrackState] = {}
        self.boundaries = None
        if road_boundaries and all(road_boundaries.get(side) for side in ("L", "R")):
            self.boundaries = {
                side: [_pixel_point(point, width, height) for point in road_boundaries[side]]
                for side in ("L", "R")
            }
        self.pair_paths = [self._pair_samples(i) for i in range(len(self.gates) - 1)]
        self.station_sections = [
            section
            for index in range(len(self.gates) - 1)
            if (section := self._make_section(
                self.gates[index], self.gates[index + 1], index
            )) is not None
        ]
        self._include_stop_line(stop_line)
        self.section_bounds = [
            (
                min(section.first.left[0], section.first.right[0],
                    section.second.left[0], section.second.right[0]),
                max(section.first.left[0], section.first.right[0],
                    section.second.left[0], section.second.right[0]),
                min(section.first.left[1], section.first.right[1],
                    section.second.left[1], section.second.right[1]),
                max(section.first.left[1], section.first.right[1],
                    section.second.left[1], section.second.right[1]),
            )
            for section in self.station_sections
        ]
        self.section_extent = (
            min((bound[0] for bound in self.section_bounds), default=0.0),
            max((bound[1] for bound in self.section_bounds), default=0.0),
            min((bound[2] for bound in self.section_bounds), default=0.0),
            max((bound[3] for bound in self.section_bounds), default=0.0),
        )
        self.regular_sections = {
            section.gate_pair_index: section
            for section in self.station_sections
            if section.first.name != "Stop line" and section.second.name != "Stop line"
        }
        if known_gate_spacing:
            self.metric_calibrator = KnownGateMetric(
                self.station_sections, self.lane_count, self.pair_paths
            )

    @staticmethod
    def _make_section(first: Gate, second: Gate, pair_index: int) -> StationSection | None:
        corners = np.float32([first.left, first.right, second.right, second.left])
        if abs(cv2.contourArea(corners)) < 2:
            return None
        source = np.float32([first.left, first.right, second.left, second.right])
        target = np.float32([
            [0, first.station_m], [1, first.station_m],
            [0, second.station_m], [1, second.station_m],
        ])
        transform = cv2.getPerspectiveTransform(source, target)
        if not np.isfinite(transform).all() or np.linalg.matrix_rank(transform) < 3:
            return None
        return StationSection(first, second, transform, pair_index)

    @staticmethod
    def _project(section: StationSection, point: Point) -> tuple[float, float, float] | None:
        x, y = point
        matrix = section.transform
        denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        if abs(denominator) < 1e-9:
            return None
        across = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
        station = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
        span = section.second.station_m - section.first.station_m
        if not isfinite(across) or not isfinite(station) or span <= 0:
            return None
        return station, across, (station - section.first.station_m) / span

    def _include_stop_line(self, stop_line: tuple[Point, Point] | None) -> None:
        if stop_line is None or not self.station_sections:
            return
        stop_midpoint = (
            (stop_line[0][0] + stop_line[1][0]) / 2,
            (stop_line[0][1] + stop_line[1][1]) / 2,
        )
        for section, prepend in (
            (self.station_sections[0], True),
            (self.station_sections[-1], False),
        ):
            projected = self._project(section, stop_midpoint)
            if projected is None:
                continue
            station = projected[0]
            span = section.second.station_m - section.first.station_m
            if prepend and section.first.station_m - 5 * span < station < section.first.station_m - 0.5:
                stop = Gate("Stop line", station, *stop_line)
                extra = self._make_section(stop, section.first, section.gate_pair_index)
                if extra is not None:
                    self.station_sections.insert(0, extra)
                return
            if not prepend and section.second.station_m + 0.5 < station < section.second.station_m + 5 * span:
                stop = Gate("Stop line", station, *stop_line)
                extra = self._make_section(section.second, stop, section.gate_pair_index)
                if extra is not None:
                    self.station_sections.append(extra)
                return

    @staticmethod
    def _extrapolate(section: StationSection, point: Point, before: bool) -> tuple[float, float] | None:
        reference = section.first if before else section.second
        other = section.second if before else section.first
        gate_vector = _subtract(reference.right, reference.left)
        gate_length = hypot(*gate_vector)
        other_center = (
            (other.left[0] + other.right[0]) / 2,
            (other.left[1] + other.right[1]) / 2,
        )
        if gate_length < 2:
            return None
        denominator = _cross(gate_vector, _subtract(other_center, reference.left))
        if abs(denominator) < gate_length * 2:
            return None
        fraction = _cross(gate_vector, _subtract(point, reference.left)) / denominator
        station = reference.station_m + fraction * (other.station_m - reference.station_m)
        across = (
            _subtract(point, reference.left)[0] * gate_vector[0]
            + _subtract(point, reference.left)[1] * gate_vector[1]
        ) / (gate_length * gate_length)
        return station, across

    def road_position(
        self, point: Point, clamp_lateral: bool = True,
    ) -> tuple[float, float, int] | None:
        """將影像接地點映到局部道路里程，並保留車道橫向比例。"""
        candidates = []
        x, y = point
        left, right, top, bottom = self.section_extent
        use_bounds = (
            len(self.station_sections) >= 16
            and left <= x <= right and top <= y <= bottom
        )
        if use_bounds:
            nearby = [
                index for index, (left, right, top, bottom) in enumerate(self.section_bounds)
                if left <= x <= right and top <= y <= bottom
            ]
        else:
            nearby = range(len(self.station_sections))

        def add_candidate(index: int) -> None:
            section = self.station_sections[index]
            projected = self._project(section, point)
            if projected is None:
                return
            station, across, fraction = projected
            outside_station = max(0.0, -fraction, fraction - 1.0)
            outside_lateral = max(0.0, -across, across - 1.0)
            score = outside_station * 2 + outside_lateral
            candidates.append((score, index, outside_station, section, station, across, fraction))

        for index in nearby:
            add_candidate(index)
        if use_bounds and (not candidates or min(item[0] for item in candidates) > 0):
            nearby_set = set(nearby)
            for index in range(len(self.station_sections)):
                if index not in nearby_set:
                    add_candidate(index)
        if not candidates:
            return None
        _, _, outside_station, section, station, across, fraction = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        if outside_station > 0:
            extrapolated = self._extrapolate(section, point, fraction < 0)
            if extrapolated is None:
                return None
            station, across = extrapolated
        elif (
            self.metric_calibrator is not None
            and section.first.name != "Stop line"
            and section.second.name != "Stop line"
        ):
            # 相鄰投影在共同 Gate 上有相同里程；在附近混合兩者，可消除導數跳躍。
            blend_band = 0.25
            neighbor_index = (
                section.gate_pair_index - 1 if fraction < blend_band
                else section.gate_pair_index + 1 if fraction > 1 - blend_band
                else None
            )
            neighbor = self.regular_sections.get(neighbor_index)
            if neighbor is not None:
                adjacent = self._project(neighbor, point)
                if adjacent is not None and abs(adjacent[0] - station) < 20:
                    edge_fraction = min(fraction, 1 - fraction)
                    weight = 0.5 * (1 - edge_fraction / blend_band)
                    station = (1 - weight) * station + weight * adjacent[0]
                    across = (1 - weight) * across + weight * adjacent[1]
        lateral = max(0.0, min(across, 1.0)) if clamp_lateral else across
        return station, lateral, section.gate_pair_index

    @staticmethod
    def _signed_line_distance(line: tuple[Point, Point], point: Point) -> float | None:
        direction = _subtract(line[1], line[0])
        length = hypot(*direction)
        if length < 2:
            return None
        return _cross(direction, _subtract(point, line[0])) / length

    def _speed_zone_margins(self, point: Point) -> tuple[float, float] | None:
        """回傳接地點朝向區間內側的停止線與末 Gate 距離。"""
        if self.speed_zone is None:
            return None
        stop_line, last_gate = self.speed_zone
        stop_midpoint = (
            (stop_line[0][0] + stop_line[1][0]) / 2,
            (stop_line[0][1] + stop_line[1][1]) / 2,
        )
        gate_midpoint = (
            (last_gate[0][0] + last_gate[1][0]) / 2,
            (last_gate[0][1] + last_gate[1][1]) / 2,
        )
        stop_reference = self._signed_line_distance(stop_line, gate_midpoint)
        gate_reference = self._signed_line_distance(last_gate, stop_midpoint)
        stop_distance = self._signed_line_distance(stop_line, point)
        gate_distance = self._signed_line_distance(last_gate, point)
        if (
            stop_reference is None or gate_reference is None
            or stop_distance is None or gate_distance is None
            or abs(stop_reference) < 2 or abs(gate_reference) < 2
        ):
            return None
        return (
            stop_distance * (1 if stop_reference > 0 else -1),
            gate_distance * (1 if gate_reference > 0 else -1),
        )

    @staticmethod
    def _reset_measurements(track: TrackState) -> None:
        track.last_crossing = None
        track.crossed_gates.clear()
        track.latest = None
        track.section_total = None
        track.section_start = None
        track.section_distance_m = 0.0
        track.section_lane_adjusted = True
        track.local_samples.clear()
        track.local_latest = None
        track.rejected_points = 0
        track.filtered_velocity_mps = None
        track.velocity_initialized = False
        track.gate_reference_velocity_mps = None
        track.gate_reference_time = None

    def _update_speed_zone(self, track: TrackState, point: Point) -> tuple[str, str]:
        """更新速度顯示區間；兩個方向都以實際進入與離開邊界判定。"""
        previous_state = track.speed_zone_state
        margins = self._speed_zone_margins(point)
        if margins is None:
            track.speed_zone_state = "inside"
        elif previous_state == "before" and min(margins) >= 0:
            self._reset_measurements(track)
            track.speed_zone_state = "inside"
        elif previous_state == "inside" and min(margins) < -2.0:
            track.speed_zone_state = "after"
        return previous_state, track.speed_zone_state

    @staticmethod
    def _smooth_gate_velocity(
        track: TrackState, raw_velocity: float, uncertainty_m: float,
        duration: float, elapsed_s: float, timestamp_s: float,
    ) -> float:
        """依遠近與接地點噪聲調整響應速度，限制單幀速度跳動。"""
        previous = track.filtered_velocity_mps
        if previous is None:
            previous = 0.0
        elapsed = max(0.0, elapsed_s)
        if (
            uncertainty_m >= 1.0
            and track.gate_reference_velocity_mps is not None
            and track.gate_reference_time is not None
            and 0 <= timestamp_s - track.gate_reference_time <= 1.0
        ):
            reference = track.gate_reference_velocity_mps
            allowance = max(4.0, abs(reference) * 0.25) + 8.0 * (
                timestamp_s - track.gate_reference_time
            )
            if reference * raw_velocity > 0 and abs(raw_velocity) > abs(reference) + allowance:
                raw_velocity = (1 if reference > 0 else -1) * (abs(reference) + allowance)
        uncertainty_m = min(uncertainty_m, 5.0) if isfinite(uncertainty_m) else 5.0
        tau = 0.12 if not track.velocity_initialized else min(0.5, 0.12 + 0.06 * uncertainty_m)
        alpha = elapsed / (tau + elapsed) if elapsed > 0 else 0.0
        change = alpha * (raw_velocity - previous)
        max_change = 12.0 * elapsed + 0.35
        filtered = previous + max(-max_change, min(change, max_change))
        track.filtered_velocity_mps = filtered
        if duration >= 0.2 and abs(raw_velocity) > 0:
            track.velocity_initialized = True
        return filtered

    def _local_estimate(
        self, track: TrackState, point: Point, timestamp_s: float
    ) -> SpeedEstimate | None:
        position = self.road_position(point)
        if position is None:
            return track.local_latest
        station, across, current_pair_index = position
        samples = track.local_samples
        calibrator = self.metric_calibrator
        pixel_uncertainty_m = (
            calibrator.station_uncertainty(current_pair_index, point)
            if isinstance(calibrator, KnownGateMetric) else 0.0
        )
        if (
            calibrator is not None and not isinstance(calibrator, KnownGateMetric)
            and len(samples) >= 3
            and samples[-1][0] - samples[0][0] >= 0.12
        ):
            prior_distances = [calibrator.metric_station(item[1]) for item in samples]
            current_m = calibrator.metric_station(station)
            if current_m is not None and all(value is not None for value in prior_distances):
                prior_times = np.asarray([item[0] for item in samples])
                prior_meters = np.asarray(prior_distances)
                # 跨多幀的中位數斜率不會被單幀框抖動放大成極端速度。
                slopes = [
                    (prior_meters[j] - prior_meters[i])
                    / (prior_times[j] - prior_times[i])
                    for i in range(len(samples))
                    for j in range(i + 1, len(samples))
                    if prior_times[j] - prior_times[i] >= 0.08
                ]
                if slopes:
                    trend = float(np.median(slopes))
                    intercept = float(np.median(prior_meters - trend * prior_times))
                    deviations = prior_meters - (intercept + trend * prior_times)
                    noise = 1.4826 * float(np.median(np.abs(deviations)))
                    elapsed = timestamp_s - samples[-1][0]
                    tolerance_m = max(
                        0.35, 0.05 * abs(trend), 3 * noise,
                        0.15 * abs(trend) * elapsed + 7.5 * elapsed ** 2,
                    )
                    if abs(current_m - (intercept + trend * timestamp_s)) > tolerance_m:
                        track.rejected_points += 1
                        if track.rejected_points >= 3:
                            samples.clear()
                            track.local_latest = None
                            track.rejected_points = 0
                            samples.append((timestamp_s, station, across))
                            return None
                        return track.local_latest
        track.rejected_points = 0
        samples.append((timestamp_s, station, across))
        # 近處的像素解析度較高，縮短回歸窗可降低加減速時的讀數延遲。
        window_s = (
            min(1.0, max(0.5, 0.45 + 0.18 * pixel_uncertainty_m))
            if isinstance(calibrator, KnownGateMetric) else 1.0
        )
        if isinstance(calibrator, KnownGateMetric) and len(samples) >= 2:
            # 低 FPS 仍保留至少四個有效位置，避免永遠停在 (estimate)。
            frame_interval = samples[-1][0] - samples[-2][0]
            window_s = max(window_s, min(3.1, 3.05 * frame_interval))
        while len(samples) > 2 and timestamp_s - samples[0][0] > window_s:
            samples.popleft()
        duration = samples[-1][0] - samples[0][0]
        if len(samples) < 2 or duration < 0.015:
            return track.local_latest
        if calibrator is not None:
            distances = [calibrator.metric_station(item[1]) for item in samples]
            if any(value is None for value in distances):
                return None
        else:
            distances = [item[1] for item in samples]
        mean_time = sum(item[0] for item in samples) / len(samples)
        mean_station = sum(item[1] for item in samples) / len(samples)
        mean_across = sum(item[2] for item in samples) / len(samples)
        denominator = sum((time - mean_time) ** 2 for time, _, _ in samples)
        if denominator <= 1e-9:
            return track.local_latest
        times = np.asarray([item[0] - mean_time for item in samples])
        metric_distances = np.asarray(distances)
        metric_mean = float(np.mean(metric_distances))
        slope = float(np.dot(times, metric_distances - metric_mean) / denominator)
        residual = metric_distances - (metric_mean + slope * times)
        residual_noise_m = (
            1.4826 * float(np.median(np.abs(residual)))
            if calibrator is not None else 0.0
        )
        if calibrator is not None and len(samples) >= 4:
            threshold = max(0.35, 3 * residual_noise_m)
            inliers = np.abs(residual) <= threshold
            if int(np.count_nonzero(inliers)) < 4:
                return track.local_latest
            fitted_time = times[inliers]
            fitted_distance = metric_distances[inliers]
            fitted_time -= float(np.mean(fitted_time))
            variance = float(np.dot(fitted_time, fitted_time))
            if variance <= 1e-9:
                return track.local_latest
            slope = float(np.dot(
                fitted_time, fitted_distance - float(np.mean(fitted_distance))
            ) / variance)
        pair_index = next(
            (
                index for index in range(len(self.gates) - 1)
                if mean_station < self.gates[index + 1].station_m
            ),
            len(self.gates) - 2,
        )
        if calibrator is not None:
            ratio = calibrator.lane_ratio(
                pair_index, *self.pair_paths[pair_index], mean_across
            )
        else:
            ratio = self.lane_distance_ratio(pair_index, mean_across)
        velocity_mps = slope * (ratio if ratio is not None else 1.0)
        speed_kmh = abs(velocity_mps) * 3.6
        if isinstance(calibrator, KnownGateMetric):
            uncertainty_m = max(pixel_uncertainty_m, residual_noise_m)
            # 透視敏感區及實際框抖動都會提高需要累積的有效位移。
            resolution_kmh = 2 * uncertainty_m / duration * 3.6
            if speed_kmh <= resolution_kmh:
                velocity_mps = 0.0
                speed_kmh = 0.0
        if not isfinite(speed_kmh) or speed_kmh > 250:
            return track.local_latest
        if isinstance(calibrator, KnownGateMetric):
            velocity_mps = self._smooth_gate_velocity(
                track, velocity_mps, uncertainty_m, duration,
                samples[-1][0] - samples[-2][0], timestamp_s,
            )
            speed_kmh = abs(velocity_mps) * 3.6
        elif calibrator is not None:
            previous = track.local_latest
            if previous is not None and speed_kmh > 0:
                elapsed = samples[-1][0] - samples[-2][0]
                alpha = min(0.5, max(0.08, elapsed / 0.5))
                speed_kmh = previous.kmh * (1 - alpha) + speed_kmh * alpha
        if speed_kmh < 1.5:
            speed_kmh = 0.0
        lane_number = min(int(across * self.lane_count), self.lane_count - 1) + 1
        track.local_latest = SpeedEstimate(
            kmh=speed_kmh,
            distance_m=speed_kmh * duration / 3.6,
            duration_s=duration,
            lane_number=lane_number,
            from_gate="",
            to_gate="",
            lane_adjusted=ratio is not None,
            method=(
                "gate_local" if isinstance(calibrator, KnownGateMetric)
                else "visual" if calibrator is not None else "local"
            ),
            confidence=(
                (calibrator.confidence if calibrator.ready else "TRACKED")
                if calibrator is not None and len(samples) >= 4 and duration >= 0.12
                and (not isinstance(calibrator, KnownGateMetric) or speed_kmh > 0 or duration >= 0.4)
                else "PROVISIONAL" if calibrator is not None else "NOMINAL"
            ),
        )
        return track.local_latest

    def _pair_samples(self, index: int) -> tuple[list[Point], list[Point]]:
        first, second = self.gates[index:index + 2]
        positions = []
        for side, first_point, second_point in (
            ("L", first.left, second.left),
            ("R", first.right, second.right),
        ):
            path = self.boundaries[side] if self.boundaries else None
            first_position = _position_on_path(path, first_point) if path else None
            second_position = _position_on_path(path, second_point) if path else None
            if first_position is None or second_position is None:
                return [first.left, second.left], [first.right, second.right]
            positions.append((path, first_position, second_position))
        samples = [
            [
                _point_on_path(
                    path,
                    first_position + (second_position - first_position) * step / 24,
                )
                for step in range(25)
            ]
            for path, first_position, second_position in positions
        ]
        return samples[0], samples[1]

    def lane_distance_ratio(self, pair_index: int, across: float) -> float | None:
        """以彎道不同橫向位置的像素路徑長度修正已知 Gate 距離。"""
        left_path, right_path = self.pair_paths[pair_index]

        def length_at(fraction: float) -> float:
            points = [
                (
                    left[0] * (1 - fraction) + right[0] * fraction,
                    left[1] * (1 - fraction) + right[1] * fraction,
                )
                for left, right in zip(left_path, right_path)
            ]
            return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))

        middle_length = length_at(0.5)
        if middle_length < 2:
            return None
        ratio = length_at(across) / middle_length
        # 過大的比例通常是透視或校正誤差；不要使用失真的車道差修正。
        return ratio if isfinite(ratio) and 0.75 <= ratio <= 1.25 else None

    def update(
        self, track_id: int, point: Point, timestamp_s: float, frame_number: int
    ) -> SpeedEstimate | None:
        track = self.tracks.setdefault(int(track_id), TrackState())
        if not isfinite(timestamp_s):
            if track.speed_zone_state != "inside":
                return None
            return track.local_latest if self.metric_calibrator is not None else track.latest
        previous = track.last_point
        previous_time = track.last_time
        previous_frame = track.last_frame
        gate_estimate = None
        if previous is not None and previous_time is not None and previous_frame is not None:
            frame_gap = frame_number - previous_frame
            if (
                frame_gap <= 0
                or frame_gap > self.max_track_gap_frames
                or timestamp_s <= previous_time
                or timestamp_s - previous_time > self.max_track_gap_seconds
            ):
                self._reset_measurements(track)
                track.speed_zone_state = "before"
                self._update_speed_zone(track, point)
            else:
                previous_zone_state, zone_state = self._update_speed_zone(track, point)
                measure_crossings = (
                    zone_state == "inside"
                    or (
                        previous_zone_state == "inside"
                        and zone_state == "after"
                    )
                )
                hits = []
                min_x, max_x = sorted((previous[0], point[0]))
                min_y, max_y = sorted((previous[1], point[1]))
                for index, gate in enumerate(self.gates):
                    if not measure_crossings:
                        break
                    if index in track.crossed_gates:
                        continue
                    left, right, top, bottom = self.gate_bounds[index]
                    if max_x < left or min_x > right or max_y < top or min_y > bottom:
                        continue
                    intersection = segment_intersection(previous, point, gate.left, gate.right)
                    if intersection is not None:
                        time_fraction, across = intersection
                        hits.append((time_fraction, index, across))
                for time_fraction, index, across in sorted(hits):
                    crossing_time = previous_time + time_fraction * (timestamp_s - previous_time)
                    prior = track.last_crossing
                    valid_pair = False
                    if prior is not None and abs(index - prior[0]) == 1:
                        previous_index, previous_crossing_time, previous_across = prior
                        duration = crossing_time - previous_crossing_time
                        lane_before = min(int(previous_across * self.lane_count), self.lane_count - 1)
                        lane_after = min(int(across * self.lane_count), self.lane_count - 1)
                        pair_index = min(previous_index, index)
                        calibrator = self.metric_calibrator
                        ratio = (
                            calibrator.lane_ratio(
                                pair_index, *self.pair_paths[pair_index],
                                (previous_across + across) / 2,
                            )
                            if calibrator is not None else
                            self.lane_distance_ratio(pair_index, (previous_across + across) / 2)
                        )
                        if lane_before == lane_after and duration >= 0.05:
                            # 車道差修正不可信時，仍以該段的尺度估算距離。
                            # 無視覺尺度時才沿用舊版的預填 Gate 距離。
                            first_station = self.gates[index].station_m
                            second_station = self.gates[previous_index].station_m
                            if calibrator is not None:
                                first_station = calibrator.metric_station(first_station)
                                second_station = calibrator.metric_station(second_station)
                            if first_station is not None and second_station is not None:
                                station_distance = abs(first_station - second_station)
                                distance = station_distance * (
                                    ratio if ratio is not None else 1.0
                                )
                                speed_kmh = distance / duration * 3.6
                                if 0 < speed_kmh <= 250:
                                    valid_pair = True
                                    track.latest = SpeedEstimate(
                                        kmh=speed_kmh,
                                        distance_m=distance,
                                        duration_s=duration,
                                        lane_number=lane_after + 1,
                                        from_gate=self.gates[previous_index].name,
                                        to_gate=self.gates[index].name,
                                        lane_adjusted=ratio is not None,
                                        confidence=(
                                            calibrator.confidence if calibrator is not None
                                            else "NOMINAL"
                                        ),
                                    )
                                    if isinstance(calibrator, KnownGateMetric):
                                        if track.section_start is None:
                                            track.section_start = prior
                                            track.section_distance_m = 0.0
                                            track.section_lane_adjusted = True
                                        track.section_distance_m += distance
                                        track.section_lane_adjusted &= ratio is not None
                                        start_index, start_time, _ = track.section_start
                                        total_duration = crossing_time - start_time
                                        track.section_total = SpeedEstimate(
                                            kmh=track.section_distance_m / total_duration * 3.6,
                                            distance_m=track.section_distance_m,
                                            duration_s=total_duration,
                                            lane_number=lane_after + 1,
                                            from_gate=self.gates[start_index].name,
                                            to_gate=self.gates[index].name,
                                            lane_adjusted=track.section_lane_adjusted,
                                            confidence=calibrator.confidence,
                                        )
                                    reference = track.section_total or track.latest
                                    track.gate_reference_velocity_mps = (
                                        (1 if index > previous_index else -1)
                                        * reference.kmh / 3.6
                                    )
                                    track.gate_reference_time = crossing_time
                                    gate_estimate = track.latest
                    if not valid_pair:
                        track.section_start = None
                        track.section_distance_m = 0.0
                        track.section_lane_adjusted = True
                    track.last_crossing = index, crossing_time, across
                    track.crossed_gates.add(index)
        else:
            self._update_speed_zone(track, point)
        track.last_point = (float(point[0]), float(point[1]))
        track.last_time = timestamp_s
        track.last_frame = frame_number
        if track.speed_zone_state != "inside":
            return None
        local_estimate = self._local_estimate(track, point, timestamp_s)
        if self.metric_calibrator is not None:
            # Gate 穿越僅作獨立校驗，不切換畫面速度公式。
            return local_estimate
        return gate_estimate or local_estimate or track.latest

    def remove(self, track_id: int) -> None:
        self.tracks.pop(int(track_id), None)

    def reassign_track(self, old_id: int, new_id: int) -> bool:
        old_id, new_id = int(old_id), int(new_id)
        if old_id not in self.tracks or new_id in self.tracks:
            return False
        self.tracks[new_id] = self.tracks.pop(old_id)
        return True

    def speed_zone_state(self, track_id: int) -> str:
        track = self.tracks.get(int(track_id))
        return track.speed_zone_state if track is not None else "before"

    def section_speed(self, track_id: int) -> SpeedEstimate | None:
        track = self.tracks.get(int(track_id))
        if track is None or track.speed_zone_state != "after":
            return None
        return track.section_total or track.latest
