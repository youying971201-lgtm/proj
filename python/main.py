from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

from Car import Vehicle
from trafficLight import TrafficLightDetector


# ==================== 基本設定 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 在這裡切換要分析的影片；下方 VIDEO_CONFIGS 會自動套用對應座標與方向。
VIDEO_NAME = "day_green_normal.mp4"
VIDEO_SOURCE = str(PROJECT_ROOT / VIDEO_NAME)
MODEL_PATH = "yolo11x.pt"
# 影片分析區間（單位：秒）。END_TIME_SECONDS 設為 None 代表處理到影片結尾。
START_TIME_SECONDS = 0.0
END_TIME_SECONDS = None
# None 代表自動選擇裝置；CUDA GPU 可填 "0"，Apple Silicon 可填 "mps"。
DEVICE = None
CLASSES_TO_DETECT = [2, 3, 5, 7, 9]
VEHICLE_CLASSES = {2, 3, 5, 7}
TRACK_TTL_FRAMES = 90
DEBUG = True
LINE_DEAD_ZONE_PIXELS = 12

# 三部影片各自的設定。座標格式為 ((起點 x, y), (終點 x, y))。
# in_transition 表示「線的哪一側移動到哪一側」算 IN，其反向則算 OUT。
# playback_speed 只控制顯示速度，不會跳過影格或降低偵測精度。
VIDEO_CONFIGS = {
    # day_green_normal.mp4（1920x1080）：車輛由遠方向鏡頭靠近，使用橫線。
    "day_green_normal.mp4": {
        "line": ((280, 650), (1120, 650)),
        "in_transition": (-1, 1),  # 畫面上方 -> 下方為 IN
        "playback_speed": 1.0,
    },
    # day_intersection_multi_car.mp4（1920x1080）：停止線前方，使用橫線。
    "day_intersection_multi_car.mp4": {
        "line": ((420, 875), (1840, 875)),
        "in_transition": (-1, 1),  # 畫面上方 -> 下方為 IN
        "playback_speed": 1.0,
    },
    # night_red_stop.mp4（1280x720）：車流左右通過畫面，使用垂直線。
    "night_red_stop.mp4": {
        "line": ((640, 345), (640, 650)),
        "in_transition": (1, -1),  # 畫面左側 -> 右側為 IN
        "playback_speed": 0.25,  # 以原始速度的 25% 顯示
    },
}


def choose_traffic_light(boxes, classes, confidences):
    """從偵測結果中選出信心分數與面積綜合最高的交通號誌。"""
    candidates = []
    for box, class_id, confidence in zip(boxes, classes, confidences):
        if class_id != 9:
            continue
        x1, y1, x2, y2 = box
        area = max((x2 - x1) * (y2 - y1), 0)
        candidates.append((float(confidence) * area, box))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def safe_roi(frame, box):
    """將邊界框限制在畫面內，避免裁切交通號誌時索引超出範圍。"""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    return frame[y1:y2, x1:x2], (int(x1), int(y1), int(x2), int(y2))


def get_video_config(video_name, width, height):
    """取得影片專用設定，並確認計數線沒有超出畫面。"""
    if video_name not in VIDEO_CONFIGS:
        available = ", ".join(VIDEO_CONFIGS)
        raise ValueError(f"影片 {video_name!r} 尚未設定。可用影片：{available}")

    config = VIDEO_CONFIGS[video_name]
    (start_x, start_y), (end_x, end_y) = config["line"]
    if not (
        0 <= start_x < width
        and 0 <= end_x < width
        and 0 <= start_y < height
        and 0 <= end_y < height
    ):
        raise ValueError(
            f"{video_name} 的計數線座標超出 {width}x{height} 畫面範圍。"
        )
    return config


def get_line_side(point, line_start, line_end):
    """計算點位於有向線段哪一側；在線附近的防抖區內回傳 0。"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    line_length = max(np.hypot(x2 - x1, y2 - y1), 1.0)
    signed_distance = ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / line_length
    if abs(signed_distance) <= LINE_DEAD_ZONE_PIXELS:
        return 0
    return 1 if signed_distance > 0 else -1


def point_is_within_line_span(point, line_start, line_end):
    """確認車輛中心投影落在線段範圍內，避免在延伸線上被錯誤計數。"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    length_squared = max(dx * dx + dy * dy, 1)
    projection = ((px - x1) * dx + (py - y1) * dy) / length_squared
    return 0.0 <= projection <= 1.0


def detect_crossing(vehicle, center, config):
    """依車輛中心跨越線段前後的側別判斷 IN／OUT，並以防抖區避免重複觸發。"""
    line_start, line_end = config["line"]
    current_side = get_line_side(center, line_start, line_end)
    if current_side == 0:
        return None
    if vehicle.line_side is None:
        vehicle.line_side = current_side
        return None
    if current_side == vehicle.line_side:
        return None

    previous_side = vehicle.line_side
    vehicle.line_side = current_side
    if not point_is_within_line_span(center, line_start, line_end):
        return None
    transition = (previous_side, current_side)
    return "IN" if transition == config["in_transition"] else "OUT"


def main():
    """開啟影片，在指定時間區間內執行號誌辨識與車輛追蹤。"""
    if START_TIME_SECONDS < 0:
        raise ValueError("START_TIME_SECONDS 不可小於 0。")
    if END_TIME_SECONDS is not None and END_TIME_SECONDS <= START_TIME_SECONDS:
        raise ValueError("END_TIME_SECONDS 必須大於 START_TIME_SECONDS。")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Cannot open video source: {VIDEO_SOURCE}")
        return

    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read the first frame: {VIDEO_SOURCE}")
        cap.release()
        return

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("無法取得影片 FPS，因此無法套用開始／結束時間設定。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if total_frames > 0 else None
    if video_duration is not None and START_TIME_SECONDS >= video_duration:
        cap.release()
        raise ValueError(
            f"開始時間 {START_TIME_SECONDS:.2f} 秒已超過影片長度 "
            f"{video_duration:.2f} 秒。"
        )

    # 使用影格定位可讓結束條件明確，並避免每一幀都依賴可能不精準的時間戳。
    start_frame = int(START_TIME_SECONDS * fps)
    end_frame = (
        int(END_TIME_SECONDS * fps) if END_TIME_SECONDS is not None else total_frames
    )
    if total_frames > 0:
        end_frame = min(end_frame, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"Video resolution: {width}x{height}")
    end_label = f"{end_frame / fps:.2f}" if end_frame > 0 else "video end"
    print(f"Processing interval: {start_frame / fps:.2f}s - {end_label}s")

    video_config = get_video_config(VIDEO_NAME, width, height)
    line_start, line_end = video_config["line"]
    playback_speed = video_config["playback_speed"]
    print(f"Counting line for {VIDEO_NAME}: start={line_start}, end={line_end}")
    print(f"Playback speed: {playback_speed:.0%}")
    light_detector = TrafficLightDetector()

    vehicles = {}
    total_vehicle_ids = set()
    illegal_vehicle_ids = set()
    crossing_events = []
    direction_counts = {"IN": 0, "OUT": 0}
    # frame_count 保留原始影片中的影格編號，方便回查違規發生位置。
    frame_count = start_frame
    traffic_light_state = "UNKNOWN"
    traffic_light_color = TrafficLightDetector.COLORS["UNKNOWN"]
    red_ratio = 0.0
    green_ratio = 0.0
    previous_light_state = "UNKNOWN"
    previous_time = perf_counter()
    smoothed_fps = 0.0

    print("Running... press 'q' to quit or 'r' to reset statistics.")

    while cap.isOpened():
        frame_started_at = perf_counter()
        # 結束時間採不包含上限的區間，例如 5~10 秒不會處理第 10 秒的影格。
        if end_frame > 0 and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        track_options = {
            "persist": True,
            "tracker": "bytetrack.yaml",
            "classes": CLASSES_TO_DETECT,
            "verbose": False,
        }
        if DEVICE is not None:
            track_options["device"] = DEVICE
        result = model.track(frame, **track_options)[0]

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            light_box = choose_traffic_light(boxes, classes, confidences)
            if light_box is not None:
                roi, (x1, y1, x2, y2) = safe_roi(frame, light_box)
                (
                    traffic_light_state,
                    traffic_light_color,
                    red_ratio,
                    green_ratio,
                ) = light_detector.update(roi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), traffic_light_color, 3)
                cv2.putText(
                    frame,
                    traffic_light_state,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    traffic_light_color,
                    2,
                )

            if traffic_light_state != previous_light_state:
                print(
                    f"Frame {frame_count}: light "
                    f"{previous_light_state} -> {traffic_light_state}"
                )
                previous_light_state = traffic_light_state

            vehicle_boxes = []
            vehicle_ids = []
            vehicle_classes = []
            vehicle_confidences = []

            for box, track_id, class_id, confidence in zip(
                boxes, ids, classes, confidences
            ):
                if class_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                vehicle_boxes.append(box)
                vehicle_ids.append(track_id)
                vehicle_classes.append(class_id)
                vehicle_confidences.append(confidence)
                total_vehicle_ids.add(int(track_id))

                if track_id not in vehicles:
                    vehicles[track_id] = Vehicle(
                        track_id=track_id,
                        last_center=center,
                        first_seen_frame=frame_count,
                        last_seen_frame=frame_count,
                    )

            if vehicle_boxes:
                for index, track_id in enumerate(vehicle_ids):
                    vehicle = vehicles[track_id]
                    x1, y1, x2, y2 = vehicle_boxes[index]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    direction = detect_crossing(vehicle, center, video_config)
                    if direction == "IN" and vehicle.crossed_in:
                        direction = None
                    elif direction == "OUT" and vehicle.crossed_out:
                        direction = None

                    if direction:
                        is_new_violation = vehicle.mark_crossing(
                            direction, traffic_light_state
                        )
                        crossing_events.append(
                            (frame_count, int(track_id), direction, traffic_light_state)
                        )
                        direction_counts[direction] += 1
                        print(
                            f"Frame {frame_count}: vehicle {track_id} crossed "
                            f"{direction}, light={traffic_light_state}"
                        )
                        if is_new_violation:
                            illegal_vehicle_ids.add(int(track_id))
                            print(f"  -> violation: vehicle {track_id}")

                    color = (0, 0, 255) if vehicle.illegal else (0, 255, 0)
                    label = "ILLEGAL" if vehicle.illegal else "OK"
                    vehicle.update_position(center, frame_count)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    cross_info = vehicle.cross_direction or ""
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {label} {cross_info}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

        # 只移除確實消失一段時間的追蹤目標，避免仍在畫面中的車輛遺失狀態。
        stale_ids = [
            track_id
            for track_id, vehicle in vehicles.items()
            if frame_count - vehicle.last_seen_frame > TRACK_TTL_FRAMES
        ]
        for track_id in stale_ids:
            del vehicles[track_id]

        now = perf_counter()
        instant_fps = 1.0 / max(now - previous_time, 1e-9)
        smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
        previous_time = now

        # 直接畫出本影片的專用計數線；端點順序也決定 IN／OUT 的側別。
        cv2.line(frame, line_start, line_end, (0, 0, 255), 4)
        cv2.circle(frame, line_start, 7, (0, 255, 255), -1)
        cv2.circle(frame, line_end, 7, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"Light: {traffic_light_state} | Frame: {frame_count} | FPS: {smoothed_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            traffic_light_color,
            2,
        )
        cv2.putText(
            frame,
            f"IN: {direction_counts['IN']} | OUT: {direction_counts['OUT']}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if DEBUG:
            cv2.putText(
                frame,
                f"Red ratio: {red_ratio:.3f} | Green ratio: {green_ratio:.3f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        cv2.imshow("Traffic Monitoring System", frame)
        # 依原始 FPS 與影片專用倍率控制播放速度；推論時間也計入每幀時間。
        target_frame_seconds = 1.0 / (fps * playback_speed)
        elapsed_seconds = perf_counter() - frame_started_at
        wait_ms = max(1, round((target_frame_seconds - elapsed_seconds) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            vehicles.clear()
            total_vehicle_ids.clear()
            illegal_vehicle_ids.clear()
            crossing_events.clear()
            direction_counts = {"IN": 0, "OUT": 0}
            light_detector.reset()
            traffic_light_state = "UNKNOWN"
            traffic_light_color = TrafficLightDetector.COLORS["UNKNOWN"]
            print("Statistics reset.")

    cap.release()
    cv2.destroyAllWindows()

    print("\n========== Summary ==========")
    print(f"Unique tracked vehicles: {len(total_vehicle_ids)}")
    print(f"Violating vehicles: {len(illegal_vehicle_ids)}")
    print(f"Crossing events: {len(crossing_events)}")
    print(f"IN: {direction_counts['IN']}")
    print(f"OUT: {direction_counts['OUT']}")
    print("=============================")


if __name__ == "__main__":
    main()
