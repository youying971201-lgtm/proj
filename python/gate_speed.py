"""以車輛接地點跨越 Gate 的時間估算區間均速。"""

from dataclasses import dataclass, field
from math import hypot, isfinite


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


@dataclass
class TrackState:
    last_point: Point | None = None
    last_time: float | None = None
    last_frame: int | None = None
    last_crossing: tuple[int, float, float] | None = None
    crossed_gates: set[int] = field(default_factory=set)
    latest: SpeedEstimate | None = None


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
        self.lane_count = int(saved_gates[0]["lane"]) if saved_gates else 1
        self.max_track_gap_frames = max_track_gap_frames
        self.tracks: dict[int, TrackState] = {}
        self.boundaries = None
        if road_boundaries and all(road_boundaries.get(side) for side in ("L", "R")):
            self.boundaries = {
                side: [_pixel_point(point, width, height) for point in road_boundaries[side]]
                for side in ("L", "R")
            }
        self.pair_paths = [self._pair_samples(i) for i in range(len(self.gates) - 1)]

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
        # 過大的比例通常是邊界、Gate 或視角校正錯誤；不要顯示看似可信的數字。
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
        if previous is not None and previous_time is not None and previous_frame is not None:
            frame_gap = frame_number - previous_frame
            if frame_gap <= 0 or frame_gap > self.max_track_gap_frames or timestamp_s <= previous_time:
                track.last_crossing = None
                track.crossed_gates.clear()
                track.latest = None
            else:
                hits = []
                for index, gate in enumerate(self.gates):
                    if index in track.crossed_gates:
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
                        if lane_before == lane_after and ratio is not None and duration >= 0.05:
                            distance = abs(self.gates[index].station_m - self.gates[previous_index].station_m) * ratio
                            speed_kmh = distance / duration * 3.6
                            if 0.5 <= speed_kmh <= 250:
                                track.latest = SpeedEstimate(
                                    kmh=speed_kmh,
                                    distance_m=distance,
                                    duration_s=duration,
                                    lane_number=lane_after + 1,
                                    from_gate=self.gates[previous_index].name,
                                    to_gate=self.gates[index].name,
                                )
                    track.last_crossing = index, crossing_time, across
                    track.crossed_gates.add(index)
        track.last_point = (float(point[0]), float(point[1]))
        track.last_time = timestamp_s
        track.last_frame = frame_number
        return track.latest

    def remove(self, track_id: int) -> None:
        self.tracks.pop(int(track_id), None)
