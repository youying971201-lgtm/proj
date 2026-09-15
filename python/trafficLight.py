from collections import Counter, deque
from typing import Deque, Tuple

import cv2
import numpy as np


DisplayColor = Tuple[int, int, int]


class TrafficLightDetector:
    """辨識紅綠燈顏色，並以多幀投票降低單幀誤判。"""

    COLORS = {
        "RED": (0, 0, 225),
        "GREEN": (0, 225, 0),
        "UNKNOWN": (225, 225, 225),
    }

    def __init__(
        self,
        history_size: int = 7,
        minimum_votes: int = 3,
        minimum_color_ratio: float = 0.02,
    ) -> None:
        # history 保存近期有效結果；stable_state 是提供給違規判斷的穩定狀態。
        self.history: Deque[str] = deque(maxlen=history_size)
        self.minimum_votes = minimum_votes
        self.minimum_color_ratio = minimum_color_ratio
        self.stable_state = "UNKNOWN"

    def detect_raw(self, roi: np.ndarray) -> Tuple[str, float, float]:
        """回傳單幀辨識結果，以及紅色、綠色像素所占比例。"""
        if roi is None or roi.size == 0:
            return "UNKNOWN", 0.0, 0.0

        # HSV 比 BGR 更容易依色相分離號誌顏色；紅色跨越色相軸兩端，需合併兩段。
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 70, 100]), np.array([15, 255, 255])),
            cv2.inRange(hsv, np.array([150, 70, 100]), np.array([180, 255, 255])),
        )
        green_mask = cv2.inRange(
            hsv, np.array([35, 50, 80]), np.array([95, 255, 255])
        )

        pixel_count = max(roi.shape[0] * roi.shape[1], 1)
        red_ratio = cv2.countNonZero(red_mask) / pixel_count
        green_ratio = cv2.countNonZero(green_mask) / pixel_count

        if red_ratio > green_ratio and red_ratio >= self.minimum_color_ratio:
            return "RED", red_ratio, green_ratio
        if green_ratio > red_ratio and green_ratio >= self.minimum_color_ratio:
            return "GREEN", red_ratio, green_ratio
        return "UNKNOWN", red_ratio, green_ratio

    def update(self, roi: np.ndarray) -> Tuple[str, DisplayColor, float, float]:
        """更新近期辨識紀錄，回傳供違規判斷使用的穩定號誌狀態。"""
        raw_state, red_ratio, green_ratio = self.detect_raw(roi)

        # 單一 UNKNOWN 影格不應清除先前可靠狀態，避免短暫遮蔽造成判斷跳動。
        if raw_state != "UNKNOWN":
            self.history.append(raw_state)
            state, votes = Counter(self.history).most_common(1)[0]
            if votes >= self.minimum_votes:
                self.stable_state = state

        return (
            self.stable_state,
            self.COLORS[self.stable_state],
            red_ratio,
            green_ratio,
        )

    def reset(self) -> None:
        """清除歷史投票與穩定狀態。"""
        self.history.clear()
        self.stable_state = "UNKNOWN"
