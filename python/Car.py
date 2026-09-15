from dataclasses import dataclass
from math import hypot
from typing import Optional, Sequence, Tuple


Point = Tuple[int, int]


@dataclass
class Vehicle:
    """保存單一追蹤車輛的位置、越線方向與違規狀態。"""

    track_id: int
    last_center: Point
    first_seen_frame: int
    last_seen_frame: int
    crossed_in: bool = False
    crossed_out: bool = False
    illegal: bool = False
    cross_direction: Optional[str] = None
    # 車輛最後一次明確位於計數線的哪一側：-1、1；在線附近時維持原值。
    line_side: Optional[int] = None
    # 車輛移動狀態。剛被追蹤時資料不足，因此先顯示 CHECKING。
    motion_state: str = "CHECKING"
    smoothed_speed: float = 0.0
    moving_frames: int = 0
    stopped_frames: int = 0

    def update_motion(
        self,
        center: Point,
        frame_number: int,
        stop_threshold: float,
        moving_threshold: float,
        confirmation_frames: int,
        smoothing_alpha: float,
        camera_transform: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """依中心點每影格位移判斷車輛是行駛中或停止。"""
        frame_gap = max(frame_number - self.last_seen_frame, 1)

        # 先用背景估計的仿射轉換，預測「車輛沒動時」中心點會被鏡頭帶到哪裡。
        expected_x, expected_y = self.last_center
        # 若追蹤曾中斷數幀，目前只有最近兩幀的鏡頭轉換，不能錯套到整段缺口。
        if camera_transform is not None and frame_gap == 1:
            expected_x = (
                camera_transform[0][0] * self.last_center[0]
                + camera_transform[0][1] * self.last_center[1]
                + camera_transform[0][2]
            )
            expected_y = (
                camera_transform[1][0] * self.last_center[0]
                + camera_transform[1][1] * self.last_center[1]
                + camera_transform[1][2]
            )

        # 實際中心點和鏡頭運動預測位置的差，才視為車輛本身的移動。
        pixel_distance = hypot(
            center[0] - expected_x,
            center[1] - expected_y,
        )
        speed = pixel_distance / frame_gap

        # 指數平滑可降低 YOLO 偵測框輕微抖動造成的狀態跳動。
        if (
            self.motion_state == "CHECKING"
            and self.moving_frames == 0
            and self.stopped_frames == 0
        ):
            self.smoothed_speed = speed
        else:
            self.smoothed_speed = (
                smoothing_alpha * speed
                + (1.0 - smoothing_alpha) * self.smoothed_speed
            )

        # 使用兩個門檻形成遲滯區間，避免速度落在臨界值時頻繁切換。
        if self.smoothed_speed <= stop_threshold:
            self.stopped_frames += 1
            self.moving_frames = 0
            if self.stopped_frames >= confirmation_frames:
                self.motion_state = "STOPPED"
        elif self.smoothed_speed >= moving_threshold:
            self.moving_frames += 1
            self.stopped_frames = 0
            if self.moving_frames >= confirmation_frames:
                self.motion_state = "MOVING"
        else:
            self.moving_frames = 0
            self.stopped_frames = 0

    #更新位置
    def update_position(
        self,
        center: Point,
        frame_number: int,
        stop_threshold: float,
        moving_threshold: float,
        confirmation_frames: int,
        smoothing_alpha: float,
        camera_transform: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.update_motion(
            center,
            frame_number,
            stop_threshold,
            moving_threshold,
            confirmation_frames,
            smoothing_alpha,
            camera_transform,
        )
        self.last_center = center
        self.last_seen_frame = frame_number
    #紀錄越線方向，只有首次在紅燈越線時回傳 True <- 這我感覺有點怪
    def mark_crossing(self, direction: str, light_state: str) -> bool:
        """記錄越線方向；"""
        if direction == "IN":
            if self.crossed_in:
                return False
            self.crossed_in = True
        elif direction == "OUT":
            if self.crossed_out:
                return False
            self.crossed_out = True
        else:
            raise ValueError(f"Unsupported crossing direction: {direction}")

        self.cross_direction = direction
        if light_state == "RED" and not self.illegal:
            self.illegal = True
            return True
        return False
