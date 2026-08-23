from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

from Car import Vehicle
from trafficLight import TrafficLightDetector


# 基本設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_NAME = "day_intersection_multi_car.mp4"
VIDEO_SOURCE = str(PROJECT_ROOT / VIDEO_NAME)
MODEL_PATH = "yolo11x.pt"
START_TIME_SECONDS = 0.0 #影片開始時間
END_TIME_SECONDS = None # 影片結束時間
DEVICE = None #顯示裝置
CLASSES_TO_DETECT = [2, 3, 5, 7, 9]
VEHICLE_CLASSES = {2, 3, 5, 7}
TRACK_TTL_FRAMES = 90
DEBUG = True
LINE_DEAD_ZONE_PIXELS = 12
# 車輛中心點的移動速度以「像素／影格」表示。
# 連續數幀低於停止門檻才算 STOPPED，高於行駛門檻才算 MOVING。
MOTION_STOP_THRESHOLD = 0.8
MOTION_MOVING_THRESHOLD = 1.5
MOTION_CONFIRM_FRAMES = 5
MOTION_SMOOTHING_ALPHA = 0.3
CAMERA_MOTION_MIN_POINTS = 12
CAMERA_MOTION_RANSAC_THRESHOLD = 2.0
CAMERA_MOTION_SCALE = 0.25
CAMERA_MOTION_MIN_INLIER_RATIO = 0.5

# in_transition 表示「線的哪一側移動到哪一側」算 IN，其反向則算 OUT。
# playback_speed=播放速度
VIDEO_CONFIGS = {
    # day_green_normal.mp4（1920x1080）：車輛由遠方向鏡頭靠近，使用橫線。
    "day_green_normal.mp4": {
        "line": ((280, 650), (1120, 650)),
        "in_transition": (-1, 1),  # 畫面上方 -> 下方為 IN
        "playback_speed": 1.0,
    },
    "day_intersection_multi_car.mp4": {
        "line": ((420, 875), (1840, 875)),
        "in_transition": (-1, 1),  # 畫面上方 -> 下方為 IN
        "playback_speed": 1.0,
    },
    "night_red_stop.mp4": {
        "line": ((640, 345), (640, 650)),
        "in_transition": (1, -1),  # 畫面左側 -> 右側為 IN
        "playback_speed": 0.25,
    },
}
#在篩選信心最高的
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

#裁切畫面防干擾
def safe_roi(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    return frame[y1:y2, x1:x2], (int(x1), int(y1), int(x2), int(y2))


def estimate_camera_motion(previous_gray, current_gray):
    """用背景特徵點估計前後影格間的攝影機平移、旋轉與縮放。"""
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if previous_gray is None:
        return identity, 0.0, 0.0

    # 光流只需估計整體背景運動，縮小影像可以大幅降低額外延遲。
    previous_small = cv2.resize(
        previous_gray,
        None,
        fx=CAMERA_MOTION_SCALE,
        fy=CAMERA_MOTION_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    current_small = cv2.resize(
        current_gray,
        None,
        fx=CAMERA_MOTION_SCALE,
        fy=CAMERA_MOTION_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    previous_points = cv2.goodFeaturesToTrack(
        previous_small,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=6,
    )
    if previous_points is None or len(previous_points) < CAMERA_MOTION_MIN_POINTS:
        return identity, 0.0, 0.0

    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_small,
        current_small,
        previous_points,
        None,
    )
    if current_points is None or status is None:
        return identity, 0.0, 0.0

    valid = status.reshape(-1).astype(bool)
    source = previous_points[valid].reshape(-1, 2)
    target = current_points[valid].reshape(-1, 2)
    if len(source) < CAMERA_MOTION_MIN_POINTS:
        return identity, 0.0, 0.0

    transform, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=CAMERA_MOTION_RANSAC_THRESHOLD,
        maxIters=1000,
        confidence=0.99,
    )
    if transform is None or not np.isfinite(transform).all():
        return identity, 0.0, 0.0

    # 縮小影像的旋轉／縮放比例相同，但平移量要換回原始解析度。
    transform = transform.astype(np.float32)
    transform[:, 2] /= CAMERA_MOTION_SCALE

    # 異常的縮放通常代表特徵配對失敗，這時寧可不補償。
    scale = float(np.hypot(transform[0, 0], transform[0, 1]))
    if not 0.9 <= scale <= 1.1:
        return identity, 0.0, 0.0

    inlier_ratio = float(inliers.mean()) if inliers is not None else 0.0
    if inlier_ratio < CAMERA_MOTION_MIN_INLIER_RATIO:
        return identity, 0.0, inlier_ratio

    height, width = current_gray.shape[:2]
    image_center = np.array([width / 2.0, height / 2.0, 1.0])
    moved_center = transform @ image_center
    camera_motion = float(np.linalg.norm(moved_center - image_center[:2]))
    return transform, camera_motion, inlier_ratio
#確認計數線沒有超出畫面
def get_video_config(video_name, width, height):
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

#用向量叉積算再現的哪一側(我不會算)->0的話是防抖區
def get_line_side(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    line_length = max(np.hypot(x2 - x1, y2 - y1), 1.0)
    signed_distance = ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / line_length
    if abs(signed_distance) <= LINE_DEAD_ZONE_PIXELS:
        return 0
    return 1 if signed_distance > 0 else -1

#確認車輛中心投影落在線段範圍內，避免在延伸線上被錯誤計數。
def point_is_within_line_span(point, line_start, line_end):
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
    previous_gray = None
    camera_motion = 0.0
    camera_motion_inlier_ratio = 0.0

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

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (
            camera_transform,
            camera_motion,
            camera_motion_inlier_ratio,
        ) = estimate_camera_motion(previous_gray, current_gray)
        previous_gray = current_gray

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
                    vehicle.update_position(
                        center,
                        frame_count,
                        MOTION_STOP_THRESHOLD,
                        MOTION_MOVING_THRESHOLD,
                        MOTION_CONFIRM_FRAMES,
                        MOTION_SMOOTHING_ALPHA,
                        camera_transform,
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    cross_info = vehicle.cross_direction or ""
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {label} {vehicle.motion_state} {cross_info}",
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
            cv2.putText(
                frame,
                f"Camera motion: {camera_motion:.2f}px | Inliers: {camera_motion_inlier_ratio:.0%}",
                (10, 130),
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
