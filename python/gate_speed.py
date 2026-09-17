"""用 Gate 校正的道路位置估算即時速度與跨 Gate 區間均速。"""

from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite

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


@dataclass(frozen=True)
class StationSection:
    first: Gate
    second: Gate
    transform: np.ndarray
    gate_pair_index: int


@dataclass
class TrackState:
    last_point: Point | None = None
    last_time: float | None = None
    last_frame: int | None = None
    last_crossing: tuple[int, float, float] | None = None
    crossed_gates: set[int] = field(default_factory=set)
    latest: SpeedEstimate | None = None
    local_samples: deque[tuple[float, float, float]] = field(default_factory=deque)
    local_latest: SpeedEstimate | None = None


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
    ) -> None:
        width, height = frame_size
        self.gates = [
            Gate(
                name=str(item["gate"]),
                station_m=float(item["station_m"]),
                left=_pixel_point(item["line"]["L"], width, height),
                right=_pixel_point(item["line"]["R"], width, height),
            )
            for item in saved_gates
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

    def road_position(self, point: Point) -> tuple[float, float, int] | None:
        """將影像接地點映到局部道路里程，並保留車道橫向比例。"""
        candidates = []
        for section in self.station_sections:
            projected = self._project(section, point)
            if projected is None:
                continue
            station, across, fraction = projected
            outside_station = max(0.0, -fraction, fraction - 1.0)
            outside_lateral = max(0.0, -across, across - 1.0)
            score = outside_station * 2 + outside_lateral
            candidates.append((score, outside_station, section, station, across, fraction))
        if not candidates:
            return None
        _, outside_station, section, station, across, fraction = min(
            candidates, key=lambda item: item[0]
        )
        if outside_station > 0:
            extrapolated = self._extrapolate(section, point, fraction < 0)
            if extrapolated is None:
                return None
            station, across = extrapolated
        return station, max(0.0, min(across, 1.0)), section.gate_pair_index

    def _local_estimate(
        self, track: TrackState, point: Point, timestamp_s: float
    ) -> SpeedEstimate | None:
        position = self.road_position(point)
        if position is None:
            return track.local_latest
        station, across, _ = position
        samples = track.local_samples
        samples.append((timestamp_s, station, across))
        while len(samples) > 2 and timestamp_s - samples[0][0] > 1.0:
            samples.popleft()
        duration = samples[-1][0] - samples[0][0]
        if len(samples) < 2 or duration < 0.12:
            return track.local_latest
        mean_time = sum(item[0] for item in samples) / len(samples)
        mean_station = sum(item[1] for item in samples) / len(samples)
        mean_across = sum(item[2] for item in samples) / len(samples)
        denominator = sum((time - mean_time) ** 2 for time, _, _ in samples)
        if denominator <= 1e-9:
            return track.local_latest
        slope = sum(
            (time - mean_time) * (distance - mean_station)
            for time, distance, _ in samples
        ) / denominator
        pair_index = next(
            (
                index for index in range(len(self.gates) - 1)
                if mean_station < self.gates[index + 1].station_m
            ),
            len(self.gates) - 2,
        )
        ratio = self.lane_distance_ratio(pair_index, mean_across)
        speed_kmh = abs(slope) * (ratio if ratio is not None else 1.0) * 3.6
        if not isfinite(speed_kmh) or speed_kmh > 250:
            return track.local_latest
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
            method="local",
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
            return track.latest
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
                track.last_crossing = None
                track.crossed_gates.clear()
                track.latest = None
                track.local_samples.clear()
                track.local_latest = None
            else:
                hits = []
                min_x, max_x = sorted((previous[0], point[0]))
                min_y, max_y = sorted((previous[1], point[1]))
                for index, gate in enumerate(self.gates):
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
                    if prior is not None and abs(index - prior[0]) == 1:
                        previous_index, previous_crossing_time, previous_across = prior
                        duration = crossing_time - previous_crossing_time
                        lane_before = min(int(previous_across * self.lane_count), self.lane_count - 1)
                        lane_after = min(int(across * self.lane_count), self.lane_count - 1)
                        pair_index = min(previous_index, index)
                        ratio = self.lane_distance_ratio(pair_index, (previous_across + across) / 2)
                        if lane_before == lane_after and duration >= 0.05:
                            # 透視可能把邊界像素弧長放大數倍；修正值不可信時
                            # 仍以操作員校正的 Gate 距離估速，避免整段沒有讀數。
                            station_distance = abs(
                                self.gates[index].station_m
                                - self.gates[previous_index].station_m
                            )
                            distance = station_distance * (
                                ratio if ratio is not None else 1.0
                            )
                            speed_kmh = distance / duration * 3.6
                            if 0.5 <= speed_kmh <= 250:
                                track.latest = SpeedEstimate(
                                    kmh=speed_kmh,
                                    distance_m=distance,
                                    duration_s=duration,
                                    lane_number=lane_after + 1,
                                    from_gate=self.gates[previous_index].name,
                                    to_gate=self.gates[index].name,
                                    lane_adjusted=ratio is not None,
                                )
                                gate_estimate = track.latest
                    track.last_crossing = index, crossing_time, across
                    track.crossed_gates.add(index)
        track.last_point = (float(point[0]), float(point[1]))
        track.last_time = timestamp_s
        track.last_frame = frame_number
        local_estimate = self._local_estimate(track, point, timestamp_s)
        return gate_estimate or local_estimate or track.latest

    def remove(self, track_id: int) -> None:
        self.tracks.pop(int(track_id), None)
