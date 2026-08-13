from dataclasses import dataclass
from typing import Optional, Tuple


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

    def update_position(self, center: Point, frame_number: int) -> None:
        """更新車輛中心點及最後出現的原始影片影格編號。"""
        self.last_center = center
        self.last_seen_frame = frame_number

    def mark_crossing(self, direction: str, light_state: str) -> bool:
        """記錄越線方向；只有首次在紅燈越線時回傳 True。"""
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
