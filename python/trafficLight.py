import cv2
import numpy as np
class trafficLight:
    def detect_traffic_light_color(roi):
        """
        輸入：紅綠燈 ROI（BGR）
        輸出：'RED' / 'GREEN' / 'UNKNOWN'
        """
        if roi is None or roi.size == 0:
            return (225,225,225)#"UNKNOWN"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 紅色（兩段）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 綠色
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
                    cv2.inRange(hsv, lower_red2, upper_red2)

        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_count = cv2.countNonZero(mask_red)
        green_count = cv2.countNonZero(mask_green)

        threshold = 100  # 面積門檻，避免雜訊

        if red_count > green_count and red_count > threshold:
            return "RED",(0,0,225)#"RED"
        elif green_count > red_count and green_count > threshold:
            return "GREEN",(0,225,0)#"GREEN"
        else:
            return "UNKNOWN",(225,225,225)#"UNKNOWN"