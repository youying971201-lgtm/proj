"""Detect lane changes by equally dividing operator-drawn curved road boundaries."""

from collections import deque
from dataclasses import asdict, dataclass
from math import hypot, isfinite
from typing import Any, Sequence

import cv2
import numpy as np


Point = tuple[float, float]
PairPath = tuple[Sequence[Point], Sequence[Point]]


def _resample_path(path: Sequence[Point], sample_count: int) -> tuple[Point, ...] | None:
    """Sample a polyline at equal normalized arc-length positions."""
    points = tuple((float(point[0]), float(point[1])) for point in path)
    if len(points) < 2 or sample_count < 2:
        return None
    segment_lengths = tuple(
        hypot(stop[0] - start[0], stop[1] - start[1])
        for start, stop in zip(points, points[1:])
    )
    total_length = sum(segment_lengths)
    if total_length < 2:
        return None

    samples = []
    segment_index = 0
    segment_start_distance = 0.0
    for index in range(sample_count):
        target = total_length * index / (sample_count - 1)
        while (
            segment_index + 1 < len(segment_lengths)
            and target > segment_start_distance + segment_lengths[segment_index]
        ):
            segment_start_distance += segment_lengths[segment_index]
            segment_index += 1
        length = segment_lengths[segment_index]
        if length <= 1e-9:
            samples.append(points[segment_index])
            continue
        fraction = max(0.0, min(1.0, (target - segment_start_distance) / length))
        start = points[segment_index]
        stop = points[segment_index + 1]
        samples.append((
            start[0] + fraction * (stop[0] - start[0]),
            start[1] + fraction * (stop[1] - start[1]),
        ))
    return tuple(samples)


def _ray_frame_scale(origin: Point, direction: Point, frame_size: tuple[int, int]) -> float | None:
    """Return the positive ray scale at the first image edge it reaches."""
    width, height = frame_size
    if width < 2 or height < 2:
        return None
    candidates = []
    for coordinate, delta, limit in (
        (origin[0], direction[0], width - 1.0),
        (origin[1], direction[1], height - 1.0),
    ):
        if delta > 1e-9:
            candidates.append((limit - coordinate) / delta)
        elif delta < -1e-9:
            candidates.append(-coordinate / delta)
    positive = [scale for scale in candidates if scale > 1e-6]
    return min(positive) if positive else None


def _extend_before_stop_line(
    left_path: tuple[Point, ...],
    right_path: tuple[Point, ...],
    lane_count: int,
    frame_size: tuple[int, int],
    extension_samples: int = 24,
) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """Extrapolate the stop-line tangent until every divider reaches an image edge."""
    if len(left_path) < 2 or len(right_path) < 2:
        return left_path, right_path
    tangent_index = min(8, len(left_path) - 1, len(right_path) - 1)
    left_start, right_start = left_path[0], right_path[0]
    left_direction = (
        left_start[0] - left_path[tangent_index][0],
        left_start[1] - left_path[tangent_index][1],
    )
    right_direction = (
        right_start[0] - right_path[tangent_index][0],
        right_start[1] - right_path[tangent_index][1],
    )
    fractions = tuple(index / lane_count for index in range(1, lane_count)) or (0.5,)
    edge_scales = []
    for fraction in fractions:
        origin = (
            left_start[0] * (1.0 - fraction) + right_start[0] * fraction,
            left_start[1] * (1.0 - fraction) + right_start[1] * fraction,
        )
        direction = (
            left_direction[0] * (1.0 - fraction) + right_direction[0] * fraction,
            left_direction[1] * (1.0 - fraction) + right_direction[1] * fraction,
        )
        scale = _ray_frame_scale(origin, direction, frame_size)
        if scale is not None:
            edge_scales.append(scale)
    if not edge_scales:
        return left_path, right_path

    # Extend far enough that every generated internal divider has crossed an
    # image edge. OpenCV clips the off-screen outer geometry while drawing.
    end_scale = max(edge_scales)
    extension_samples = max(2, int(extension_samples))
    extended_left = []
    extended_right = []
    for index in range(extension_samples, 0, -1):
        scale = end_scale * index / extension_samples
        extended_left.append((
            left_start[0] + scale * left_direction[0],
            left_start[1] + scale * left_direction[1],
        ))
        extended_right.append((
            right_start[0] + scale * right_direction[0],
            right_start[1] + scale * right_direction[1],
        ))
    return (
        tuple(extended_left) + left_path,
        tuple(extended_right) + right_path,
    )


@dataclass(frozen=True)
class LaneLayout:
    """Piecewise curved lane geometry derived from the two outer road boundaries."""

    ready: bool
    lane_count: int
    separators_at_gates: tuple[tuple[float, ...], ...] = ()
    gate_stations: tuple[float, ...] = ()
    message: str = ""
    display_polylines: tuple[tuple[Point, ...], ...] = ()
    display_boundaries: tuple[tuple[Point, ...], ...] = ()
    pair_paths: tuple[tuple[tuple[Point, ...], tuple[Point, ...]], ...] = ()

    def separators(self, station: float, pair_index: int) -> tuple[float, ...] | None:
        if not self.ready or not self.separators_at_gates:
            return None
        index = max(0, min(int(pair_index), len(self.separators_at_gates) - 1))
        return self.separators_at_gates[index]

    @staticmethod
    def _project_quad(point: Point, corners: np.ndarray) -> tuple[float, float] | None:
        if abs(cv2.contourArea(corners)) < 2:
            return None
        target = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
        matrix = cv2.getPerspectiveTransform(corners, target)
        x, y = point
        denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        if abs(denominator) < 1e-9:
            return None
        across = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
        along = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
        if not isfinite(across) or not isfinite(along):
            return None
        return float(across), float(along)

    def curved_across(self, point: Point, pair_hint: int) -> float | None:
        """Map a point through the closest small boundary quad instead of one straight road quad."""
        if not self.ready or not self.pair_paths:
            return None
        pair_hint = max(0, min(int(pair_hint), len(self.pair_paths) - 1))
        pair_indices = [pair_hint]
        if pair_hint > 0:
            pair_indices.append(pair_hint - 1)
        if pair_hint + 1 < len(self.pair_paths):
            pair_indices.append(pair_hint + 1)

        candidates = []
        for pair_index in pair_indices:
            left_path, right_path = self.pair_paths[pair_index]
            for step, (left_a, left_b, right_a, right_b) in enumerate(zip(
                left_path, left_path[1:], right_path, right_path[1:]
            )):
                corners = np.float32([left_a, right_a, left_b, right_b])
                projected = self._project_quad(point, corners)
                if projected is None:
                    continue
                _, along = projected
                sample_along = max(0.0, min(1.0, along))
                left = (
                    left_a[0] + sample_along * (left_b[0] - left_a[0]),
                    left_a[1] + sample_along * (left_b[1] - left_a[1]),
                )
                right = (
                    right_a[0] + sample_along * (right_b[0] - right_a[0]),
                    right_a[1] + sample_along * (right_b[1] - right_a[1]),
                )
                cross_x = right[0] - left[0]
                cross_y = right[1] - left[1]
                width_squared = cross_x * cross_x + cross_y * cross_y
                if width_squared < 4:
                    continue
                across = (
                    (point[0] - left[0]) * cross_x
                    + (point[1] - left[1]) * cross_y
                ) / width_squared
                projected_x = left[0] + across * cross_x
                projected_y = left[1] + across * cross_y
                normalized_error = hypot(
                    point[0] - projected_x,
                    point[1] - projected_y,
                ) / max(width_squared ** 0.5, 1.0)
                outside_along = max(0.0, -along, along - 1.0)
                outside_across = max(0.0, -across, across - 1.0)
                pair_penalty = 0.03 * abs(pair_index - pair_hint)
                score = (
                    2.0 * outside_along
                    + outside_across
                    + 0.1 * normalized_error
                    + pair_penalty
                )
                candidates.append((score, pair_index, step, across))
        if not candidates:
            return None
        score, _, _, across = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        if score > 0.35 or not -0.25 <= across <= 1.25:
            return None
        return across

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "lane_count": self.lane_count,
            "separator_count": max(self.lane_count - 1, 0) if self.ready else 0,
            "outer_boundary_count": len(self.display_boundaries) if self.ready else 0,
            "source": "boundary_equal",
            "message": self.message,
        }


def build_equal_lane_layout(
    gates: Sequence[Any],
    pair_paths: Sequence[PairPath],
    lane_count: int,
    *,
    boundary_paths: dict[str, Sequence[Point]] | None = None,
    frame_size: tuple[int, int] | None = None,
) -> LaneLayout:
    """Create equal-width curved lanes at every sampled boundary cross-section."""
    lane_count = max(int(lane_count), 1)
    if len(gates) < 2 or len(pair_paths) != len(gates) - 1:
        return LaneLayout(False, lane_count, message="道路邊界或 Gate 資料不足")

    normalized_paths = []
    for left_path, right_path in pair_paths:
        left = tuple((float(point[0]), float(point[1])) for point in left_path)
        right = tuple((float(point[0]), float(point[1])) for point in right_path)
        if len(left) < 2 or len(left) != len(right):
            return LaneLayout(False, lane_count, message="左右道路邊界無法逐段配對")
        normalized_paths.append((left, right))

    geometry_paths = normalized_paths
    uses_full_boundaries = False
    if boundary_paths is not None:
        left_boundary = boundary_paths.get("L", ())
        right_boundary = boundary_paths.get("R", ())
        # Keep approximately the same number of small projection quads as the
        # former Gate-to-Gate geometry, but distribute them over the complete
        # drawn boundaries: stop line to the final boundary endpoints.
        sample_count = max(25, 24 * max(len(normalized_paths), 1) + 1)
        sampled_left = _resample_path(left_boundary, sample_count)
        sampled_right = _resample_path(right_boundary, sample_count)
        if sampled_left is None or sampled_right is None:
            return LaneLayout(False, lane_count, message="完整左右道路邊界取樣失敗")
        if frame_size is not None:
            sampled_left, sampled_right = _extend_before_stop_line(
                sampled_left,
                sampled_right,
                lane_count,
                frame_size,
            )
        geometry_paths = [(sampled_left, sampled_right)]
        uses_full_boundaries = True

    fractions = tuple(index / lane_count for index in range(1, lane_count))
    left_display = []
    right_display = []
    for pair_index, (left_path, right_path) in enumerate(geometry_paths):
        left_points = list(left_path)
        right_points = list(right_path)
        if pair_index > 0:
            left_points = left_points[1:]
            right_points = right_points[1:]
        left_display.extend(left_points)
        right_display.extend(right_points)
    display_polylines = []
    for fraction in fractions:
        polyline = []
        for pair_index, (left_path, right_path) in enumerate(geometry_paths):
            pair_points = [
                (
                    left[0] * (1.0 - fraction) + right[0] * fraction,
                    left[1] * (1.0 - fraction) + right[1] * fraction,
                )
                for left, right in zip(left_path, right_path)
            ]
            if pair_index > 0 and pair_points:
                pair_points = pair_points[1:]
            polyline.extend(pair_points)
        if len(polyline) < 2:
            return LaneLayout(False, lane_count, message="彎曲車道分隔線取樣不足")
        display_polylines.append(tuple(polyline))

    # Each sample uses the same fraction between L and R, so every segment remains
    # equally spaced from the two curved edges even when road width changes.
    return LaneLayout(
        True,
        lane_count,
        separators_at_gates=tuple(fractions for _ in gates),
        gate_stations=tuple(float(gate.station_m) for gate in gates),
        message=(
            f"已由道路邊界末端經停止線延伸至畫面邊緣，建立 {lane_count} 個車道"
            if uses_full_boundaries and frame_size is not None else
            f"已從停止線至道路邊界末端建立 {lane_count} 個車道及外邊界"
            if uses_full_boundaries else
            f"已依左右彎曲邊界建立 {lane_count} 個車道及外邊界"
        ),
        display_polylines=tuple(display_polylines),
        display_boundaries=(tuple(left_display), tuple(right_display)),
        pair_paths=tuple(geometry_paths),
    )


@dataclass(frozen=True)
class LaneChangeEvent:
    track_id: int
    direction: str
    from_lane: int
    to_lane: int
    frame_number: int
    timestamp_s: float | None
    outside: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LaneObservation:
    lane_number: int | None
    state: str
    direction: str | None = None
    from_lane: int | None = None
    to_lane: int | None = None
    event: LaneChangeEvent | None = None
    outside_side: str | None = None

    @property
    def overlay_label(self) -> str:
        if self.state == "OUTSIDE" and self.outside_side:
            arrow = "L" if self.outside_side == "LEFT" else "R"
            return f"OUT-{arrow}"
        if self.lane_number is None:
            return "LANE ?"
        if (
            self.direction
            and self.from_lane is not None
            and self.to_lane is not None
        ):
            prefix = "LC" if self.state == "CHANGING" else "CHANGED"
            arrow = "L" if self.direction == "LEFT" else "R"
            destination = "OUT" if self.outside_side else str(self.to_lane)
            return f"{prefix}-{arrow} {self.from_lane}>{destination}"
        return f"L{self.lane_number}"


@dataclass
class _TrackState:
    stable_lane: int | None = None
    initial_lane: int | None = None
    initial_frames: int = 0
    candidate_lane: int | None = None
    candidate_frames: int = 0
    last_frame: int | None = None
    last_event_frame: int | None = None
    last_direction: str | None = None
    last_from_lane: int | None = None
    last_to_lane: int | None = None
    outside_side: str | None = None


class LaneChangeDetector:
    def __init__(
        self,
        layout: LaneLayout,
        *,
        initial_confirm_frames: int = 3,
        change_confirm_frames: int = 6,
        hysteresis_ratio: float = 0.08,
        max_track_gap_frames: int = 5,
        changed_label_frames: int = 30,
        recent_event_limit: int = 20,
    ) -> None:
        self.layout = layout
        self.initial_confirm_frames = max(1, int(initial_confirm_frames))
        self.change_confirm_frames = max(1, int(change_confirm_frames))
        self.hysteresis_ratio = max(0.0, min(float(hysteresis_ratio), 0.25))
        self.max_track_gap_frames = max(1, int(max_track_gap_frames))
        self.changed_label_frames = max(0, int(changed_label_frames))
        self.tracks: dict[int, _TrackState] = {}
        self.recent_events: deque[LaneChangeEvent] = deque(maxlen=recent_event_limit)
        self.left_count = 0
        self.right_count = 0

    @staticmethod
    def _lane_of(across: float, separators: Sequence[float]) -> int:
        if across < 0:
            return 0
        if across > 1:
            return len(separators) + 2
        return 1 + sum(across > boundary for boundary in separators)

    @staticmethod
    def _direction(from_lane: int, to_lane: int) -> str:
        return "LEFT" if to_lane < from_lane else "RIGHT"

    def _observation(self, state: _TrackState, frame_number: int, event=None) -> LaneObservation:
        if state.stable_lane is None:
            return LaneObservation(None, "CHECKING")
        if state.candidate_lane is not None:
            direction = self._direction(state.stable_lane, state.candidate_lane)
            outside = not 1 <= state.candidate_lane <= self.layout.lane_count
            return LaneObservation(
                state.stable_lane,
                "CHANGING",
                direction,
                state.stable_lane,
                state.candidate_lane,
                event,
                direction if outside else None,
            )
        if state.last_event_frame is not None and frame_number - state.last_event_frame <= self.changed_label_frames:
            outside = not 1 <= (state.last_to_lane or 0) <= self.layout.lane_count
            return LaneObservation(
                state.stable_lane, "CHANGED", state.last_direction,
                state.last_from_lane, state.last_to_lane, event,
                state.last_direction if outside else None,
            )
        if state.outside_side is not None:
            return LaneObservation(
                state.stable_lane,
                "OUTSIDE",
                direction=state.outside_side,
                outside_side=state.outside_side,
            )
        return LaneObservation(state.stable_lane, "KEEP")

    def update(
        self,
        track_id: int,
        road_position: tuple[float, float, int] | None,
        frame_number: int,
        timestamp_s: float | None = None,
        point: Point | None = None,
    ) -> LaneObservation:
        if not self.layout.ready or road_position is None:
            return LaneObservation(None, "DISABLED" if not self.layout.ready else "OUTSIDE")
        station, across, pair_index = road_position
        if point is not None:
            curved_across = self.layout.curved_across(point, pair_index)
            if curved_across is None:
                return LaneObservation(None, "OUTSIDE")
            across = curved_across
        if not all(isfinite(value) for value in (station, across)) or not -0.25 <= across <= 1.25:
            return LaneObservation(None, "OUTSIDE")
        separators = self.layout.separators(station, pair_index)
        if separators is None:
            return LaneObservation(None, "OUTSIDE")
        across = float(across)
        raw_lane = self._lane_of(across, separators)
        track_id = int(track_id)
        state = self.tracks.setdefault(track_id, _TrackState())
        if state.last_frame is not None and frame_number - state.last_frame > self.max_track_gap_frames:
            state = _TrackState()
            self.tracks[track_id] = state
        if state.last_frame is not None and frame_number <= state.last_frame:
            return self._observation(state, frame_number)
        state.last_frame = int(frame_number)

        # A track must first be confirmed in a real lane. A vehicle that first
        # appears outside the drawn road must not establish synthetic lane 0 or
        # lane_count + 1 as its stable lane; doing so would also make later
        # separator indexing invalid on a one-lane road.
        if state.stable_lane is not None and not (
            1 <= state.stable_lane <= self.layout.lane_count
        ):
            state.stable_lane = None
            state.initial_lane = None
            state.initial_frames = 0
            state.candidate_lane = None
            state.candidate_frames = 0

        if state.outside_side is not None:
            still_outside = (
                raw_lane == 0 if state.outside_side == "LEFT"
                else raw_lane == self.layout.lane_count + 1
            )
            if still_outside:
                state.initial_lane = None
                state.initial_frames = 0
                return self._observation(state, frame_number)
            if 1 <= raw_lane <= self.layout.lane_count:
                if state.initial_lane == raw_lane:
                    state.initial_frames += 1
                else:
                    state.initial_lane = raw_lane
                    state.initial_frames = 1
                if state.initial_frames >= self.initial_confirm_frames:
                    state.stable_lane = raw_lane
                    state.outside_side = None
                    state.initial_lane = None
                    state.initial_frames = 0
                    state.last_event_frame = None
                    state.last_direction = None
                    state.last_from_lane = None
                    state.last_to_lane = None
                return self._observation(state, frame_number)
            return self._observation(state, frame_number)

        if state.stable_lane is None:
            if not 1 <= raw_lane <= self.layout.lane_count:
                state.initial_lane = None
                state.initial_frames = 0
                outside_side = "LEFT" if raw_lane == 0 else "RIGHT"
                return LaneObservation(
                    None,
                    "OUTSIDE",
                    direction=outside_side,
                    outside_side=outside_side,
                )
            if state.initial_lane == raw_lane:
                state.initial_frames += 1
            else:
                state.initial_lane = raw_lane
                state.initial_frames = 1
            if state.initial_frames >= self.initial_confirm_frames:
                state.stable_lane = raw_lane
                state.initial_lane = None
                state.initial_frames = 0
            return self._observation(state, frame_number)

        target = None
        if raw_lane > state.stable_lane:
            if state.stable_lane == self.layout.lane_count:
                boundary = 1.0
                right_width = 1.0 - (separators[-1] if separators else 0.0)
            else:
                boundary = separators[state.stable_lane - 1]
                # The destination lane may be the rightmost lane. In that case
                # its right edge is the outer road boundary, not another
                # internal separator (which does not exist for a two-lane road).
                right_edge = (
                    separators[state.stable_lane]
                    if state.stable_lane < len(separators)
                    else 1.0
                )
                right_width = right_edge - boundary
            if across >= boundary + self.hysteresis_ratio * right_width:
                target = raw_lane
        elif raw_lane < state.stable_lane:
            if state.stable_lane == 1:
                boundary = 0.0
                left_width = separators[0] if separators else 1.0
            else:
                boundary = separators[state.stable_lane - 2]
                left_width = boundary - (
                    separators[state.stable_lane - 3] if state.stable_lane > 2 else 0.0
                )
            if across <= boundary - self.hysteresis_ratio * left_width:
                target = raw_lane

        if target is None:
            state.candidate_lane = None
            state.candidate_frames = 0
            return self._observation(state, frame_number)
        if state.candidate_lane == target:
            state.candidate_frames += 1
        else:
            state.candidate_lane = target
            state.candidate_frames = 1
        if state.candidate_frames < self.change_confirm_frames:
            return self._observation(state, frame_number)

        from_lane = state.stable_lane
        direction = self._direction(from_lane, target)
        outside = not 1 <= target <= self.layout.lane_count
        event = LaneChangeEvent(
            track_id, direction, from_lane, target, int(frame_number),
            float(timestamp_s) if timestamp_s is not None else None,
            outside,
        )
        if outside:
            state.outside_side = direction
        else:
            state.stable_lane = target
        state.candidate_lane = None
        state.candidate_frames = 0
        state.last_event_frame = int(frame_number)
        state.last_direction = direction
        state.last_from_lane = from_lane
        state.last_to_lane = target
        self.recent_events.append(event)
        if direction == "LEFT":
            self.left_count += 1
        else:
            self.right_count += 1
        return self._observation(state, frame_number, event)

    def remove(self, track_id: int) -> None:
        self.tracks.pop(int(track_id), None)

    def as_dict(self) -> dict:
        return {
            **self.layout.as_dict(),
            "enabled": self.layout.ready,
            "total": self.left_count + self.right_count,
            "left": self.left_count,
            "right": self.right_count,
            "recent_events": [event.as_dict() for event in reversed(self.recent_events)],
        }
