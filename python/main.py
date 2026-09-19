import argparse
import json
import math
import os
import signal
from pathlib import Path
from threading import Condition, Lock, Thread
from time import perf_counter

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse


# 直接執行 main.py 時也把 Ultralytics 設定保存在專案內，不依賴 .bat 設定環境變數。
_DEFAULT_YOLO_CONFIG_DIR = Path(__file__).resolve().parent.parent / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_YOLO_CONFIG_DIR))
(_DEFAULT_YOLO_CONFIG_DIR / "Ultralytics").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

from Car import Vehicle
from gate_speed import GateSpeedEstimator, known_gate_stations, segment_intersection
from road_marking import RoadMarkingDetector, draw_road_markings
from trafficLight import TrafficLightDetector
from video_source import CAMERA_BACKENDS, list_available_cameras, open_video_capture

# 基本設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 影像來源預設值：改成 "camera" 使用攝影機，改成 "video" 使用下方影片。
DEFAULT_INPUT_MODE = "video"
DEFAULT_VIDEO_NAME = "newVid1.MOV"
DEFAULT_CAMERA_INDEX = 1
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30.0
# 預設 YOLO 負責行人、交通號誌及依前端模式選定的汽車類別。
MODEL_PATH = "yolo11x.pt"
OFFICIAL_YOLO_MODELS = {
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
}
VEHICLE_MODEL_PATH = PROJECT_ROOT / "models" / "my_model.pt"
ROAD_MARKING_MODEL_PATH = (
    PROJECT_ROOT / "models" / "road_marking" / "stage5_roadmark_best.pth"
)
START_TIME_SECONDS = 0.0 #影片開始時間
END_TIME_SECONDS = None # 影片結束時間
# YOLO11 COCO 類別：person=0、car=2、motorcycle=3、bus=5、truck=7、traffic light=9。
PERSON_CLASS = 0
TRAFFIC_LIGHT_CLASS = 9
ENTERING_PRIMARY_VEHICLE_CLASSES = {2, 5, 7}
ENTERING_PRIMARY_CLASSES = [0, 2, 5, 7, 9]
LEAVING_PRIMARY_CLASSES = [0, 9]
# models/my_model.pt 類別：car=2、scooter=4。
CUSTOM_VEHICLE_CLASSES = {2, 4}
ENTERING_CUSTOM_VEHICLE_CLASSES = {4}
DETECTION_MODE_ENTERING = "entering"
DETECTION_MODE_LEAVING = "leaving"
DETECTION_MODES = {DETECTION_MODE_ENTERING, DETECTION_MODE_LEAVING}
MPS_IMAGE_SIZE = 640  # M2 Air GPU 上兼顧偵測品質與長時間運行溫度／速度。
CPU_IMAGE_SIZE = 512  # CPU／CUDA 共用；CPU 上避免 YOLO11x 過度延遲。
TRACK_TTL_FRAMES = 90
DEBUG = True
LINE_DEAD_ZONE_PIXELS = 12
# 車輛中心點的移動速度以「像素／影格」表示。
# 連續數幀低於停止門檻才算 STOPPED，高於行駛門檻才算 MOVING。
MOTION_STOP_THRESHOLD = 0.5
MOTION_MOVING_THRESHOLD = 1.5
MOTION_CONFIRM_FRAMES = 5
MOTION_SMOOTHING_ALPHA = 0.3
CAMERA_MOTION_MIN_POINTS = 12
CAMERA_MOTION_RANSAC_THRESHOLD = 2.0
CAMERA_MOTION_SCALE = 0.25
CAMERA_MOTION_MIN_INLIER_RATIO = 0.5

# 網頁端設定的判斷線會保存於此檔案；程式下次啟動時也會自動讀回。
POINTS_FILE = PROJECT_ROOT / "cord_data" / "points_data.json"


class FrameBroadcaster:
    """保存最新一張處理完成的 JPEG，供多個 MJPEG 瀏覽器連線讀取。"""

    def __init__(self):
        self._condition = Condition()
        self._frame = None
        self._sequence = 0
        self._finished = False

    def publish(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        with self._condition:
            self._frame = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def frames(self):
        sequence = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._sequence != sequence or self._finished,
                    timeout=1.0,
                )
                if self._sequence == sequence and self._finished:
                    return
                if self._sequence == sequence or self._frame is None:
                    continue
                sequence = self._sequence
                frame = self._frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    def finish(self):
        with self._condition:
            self._finished = True
            self._condition.notify_all()

    def reset(self):
        with self._condition:
            self._finished = False
            self._sequence += 1
            self._condition.notify_all()

    def latest_frame(self):
        with self._condition:
            return self._frame


def detection_plan(detection_mode):
    """回傳預設 YOLO 類別，以及兩個模型各自提供的車輛類別。"""
    if detection_mode == DETECTION_MODE_ENTERING:
        return (
            ENTERING_PRIMARY_CLASSES,
            ENTERING_PRIMARY_VEHICLE_CLASSES,
            ENTERING_CUSTOM_VEHICLE_CLASSES,
        )
    if detection_mode == DETECTION_MODE_LEAVING:
        return LEAVING_PRIMARY_CLASSES, set(), CUSTOM_VEHICLE_CLASSES
    raise ValueError(f"不支援的主要偵測模式：{detection_mode}")


class FallbackVehicleTracker:
    """ByteTrack 暫無 ID 時，用相鄰影格的框延續短期車輛身分。"""

    def __init__(self, max_gap_frames=5):
        self.max_gap_frames = max_gap_frames
        self.next_id = -1
        self.tracks = {}
        self.formal_tracks = {}

    @staticmethod
    def _match_score(previous, current):
        x1 = max(previous[0], current[0])
        y1 = max(previous[1], current[1])
        x2 = min(previous[2], current[2])
        y2 = min(previous[3], current[3])
        overlap = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, previous[2] - previous[0]) * max(0, previous[3] - previous[1])
        second_area = max(0, current[2] - current[0]) * max(0, current[3] - current[1])
        union = first_area + second_area - overlap
        iou = overlap / union if union > 0 else 0.0
        first_center = ((previous[0] + previous[2]) / 2, previous[3])
        second_center = ((current[0] + current[2]) / 2, current[3])
        distance = math.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )
        scale = max(
            previous[2] - previous[0], previous[3] - previous[1],
            current[2] - current[0], current[3] - current[1], 1,
        )
        if iou < 0.1 and distance > 0.6 * scale:
            return None
        return iou - 0.2 * distance / scale

    def assign(self, boxes, frame_number):
        self.tracks = {
            track_id: (box, seen)
            for track_id, (box, seen) in self.tracks.items()
            if frame_number - seen <= self.max_gap_frames
        }
        used = set()
        assigned = []
        for box in boxes:
            best = None
            for track_id, (previous, _) in self.tracks.items():
                if track_id in used:
                    continue
                score = self._match_score(previous, box)
                if score is not None and (best is None or score > best[0]):
                    best = score, track_id
            if best is None:
                track_id = self.next_id
                self.next_id -= 1
            else:
                track_id = best[1]
            used.add(track_id)
            self.tracks[track_id] = (tuple(box), frame_number)
            assigned.append(track_id)
        return assigned

    def claim_formal_ids(
        self, boxes, ids, frame_number, active_temporary_ids, known_ids,
    ):
        """短暫無 ID 的同車取得正式 ID 時，保守地配對原測速軌跡。"""
        active_temporary_ids = set(active_temporary_ids)
        known_ids = set(known_ids)
        candidates = []
        for box, formal_id in zip(boxes, ids):
            if formal_id < 0 or formal_id in known_ids:
                continue
            for temporary_id, (previous, seen) in self.tracks.items():
                if (
                    temporary_id in active_temporary_ids
                    or temporary_id not in known_ids
                    or not 1 <= frame_number - seen <= 2
                ):
                    continue
                score = self._match_score(previous, box)
                if score is not None and score >= 0.15:
                    candidates.append((score, temporary_id, formal_id))
        claimed_temporary = set()
        claimed_formal = set()
        matches = []
        for _, temporary_id, formal_id in sorted(candidates, reverse=True):
            if temporary_id in claimed_temporary or formal_id in claimed_formal:
                continue
            claimed_temporary.add(temporary_id)
            claimed_formal.add(formal_id)
            matches.append((temporary_id, formal_id))
        for temporary_id, _ in matches:
            del self.tracks[temporary_id]
        return matches

    def claim_temporary_ids(
        self, boxes, ids, frame_number, active_formal_ids, known_ids,
    ):
        """正式 ID 暫時消失時，將同一台車的測速狀態交給 TEMP。"""
        active_formal_ids = set(active_formal_ids)
        known_ids = set(known_ids)
        candidates = []
        for box, temporary_id in zip(boxes, ids):
            if temporary_id >= 0 or temporary_id in known_ids:
                continue
            for formal_id, (previous, seen) in self.formal_tracks.items():
                if (
                    formal_id in active_formal_ids
                    or formal_id not in known_ids
                    or not 1 <= frame_number - seen <= 2
                ):
                    continue
                score = self._match_score(previous, box)
                if score is not None and score >= 0.15:
                    candidates.append((score, formal_id, temporary_id))
        claimed_formal = set()
        claimed_temporary = set()
        matches = []
        for _, formal_id, temporary_id in sorted(candidates, reverse=True):
            if formal_id in claimed_formal or temporary_id in claimed_temporary:
                continue
            claimed_formal.add(formal_id)
            claimed_temporary.add(temporary_id)
            matches.append((formal_id, temporary_id))
        return matches

    def observe_formal_ids(self, boxes, ids, frame_number):
        self.formal_tracks = {
            track_id: (box, seen)
            for track_id, (box, seen) in self.formal_tracks.items()
            if frame_number - seen <= self.max_gap_frames
        }
        for box, track_id in zip(boxes, ids):
            if track_id >= 0:
                self.formal_tracks[track_id] = (tuple(box), frame_number)


def tracked_vehicle_detections(result, accepted_classes, track_namespace=None):
    """從單一模型結果取出指定車種，並保留框、追蹤 ID 與遮罩。"""
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    raw_ids = (
        result.boxes.id.cpu().numpy().astype(int)
        if result.boxes.id is not None else None
    )
    mask_polygons = getattr(getattr(result, "masks", None), "xy", None)
    detections = []
    for index, (box, class_id) in enumerate(zip(boxes, classes)):
        if class_id not in accepted_classes:
            continue
        raw_id = int(raw_ids[index]) if raw_ids is not None else -1
        if raw_id >= 0 and track_namespace is not None:
            raw_id = raw_id * 2 + track_namespace
        polygon = (
            mask_polygons[index]
            if mask_polygons is not None and index < len(mask_polygons)
            else None
        )
        detections.append((box, raw_id, polygon))
    return detections


def merge_vehicle_detections(primary_detections, custom_detections):
    """合併兩個模型的車框；重疊時保留 my_model.pt 的 scooter。"""
    def overlap_over_smaller(first, second):
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        smaller_area = min(first_area, second_area)
        return intersection / smaller_area if smaller_area > 0 else 0.0

    filtered_primary = [
        detection
        for detection in primary_detections
        if not any(
            overlap_over_smaller(detection[0], custom_detection[0]) >= 0.6
            for custom_detection in custom_detections
        )
    ]
    return filtered_primary + custom_detections


def draw_pedestrians(frame, boxes, classes, confidences):
    """只繪製行人框與信心分數，不建立追蹤、測速或事件狀態。"""
    for box, class_id, confidence in zip(boxes, classes, confidences):
        if class_id != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        color = (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"PERSON {float(confidence):.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def vehicle_ground_contact(box, polygon=None):
    """有實例輪廓時取底部接觸區；一般 YOLO 框則用底部中心。"""
    x1, _, x2, y2 = box
    if polygon is not None:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(points) >= 3 and np.isfinite(points).all():
            bottom_y = float(points[:, 1].max())
            near_bottom = points[points[:, 1] >= bottom_y - 0.04 * max(y2 - box[1], 1)]
            if len(near_bottom):
                contact_x = float(np.median(near_bottom[:, 0]))
                if x1 <= contact_x <= x2 and abs(bottom_y - y2) <= 0.15 * max(y2 - box[1], 1):
                    return round(contact_x), round(bottom_y)
    return (round((x1 + x2) / 2), round(y2))


frame_broadcaster = FrameBroadcaster()
line_lock = Lock()
runtime_status = {"state": "waiting", "message": "等待判斷程式啟動"}
road_marking_status = {
    "enabled": False,
    "state": "disabled",
    "message": "道路標線辨識尚未啟用",
    "result": None,
}

# in_transition 表示「線的哪一側移動到哪一側」算 IN，其反向則算 OUT。
VIDEO_CONFIGS = {
    # day_green_normal.mp4（1920x1080）：車輛由遠方向鏡頭靠近，使用橫線。
    "day_green_normal.mp4": {
        "line": ((280, 650), (1120, 650)),
        "in_transition": (-1, 1),  # 畫面上方 -> 下方為 IN
    },
    "day_intersection_multi_car.mp4": {
        "line": ((420, 875), (1840, 875)),
        "in_transition": (-1, 1),  # 畫面上方 -> 下方為 IN
    },
    "night_red_stop.mp4": {
        "line": ((640, 345), (640, 650)),
        "in_transition": (1, -1),  # 畫面左側 -> 右側為 IN
    },
}

#在篩選信心最高的
def choose_traffic_light(boxes, classes, confidences):
    """從偵測結果中選出信心分數與面積綜合最高的交通號誌。"""
    candidates = []
    for box, class_id, confidence in zip(boxes, classes, confidences):
        if class_id != TRAFFIC_LIGHT_CLASS:
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
        line_y = round(height * 0.7)
        return {
            "line": ((round(width * 0.15), line_y), (round(width * 0.85), line_y)),
            "in_transition": (-1, 1),
        }

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

#用向量叉積算再現的哪一側->0的話是防抖區
def get_line_side(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    line_length = max(np.hypot(x2 - x1, y2 - y1), 1.0)
    signed_distance = ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / line_length
    if abs(signed_distance) <= LINE_DEAD_ZONE_PIXELS:
        return 0
    return 1 if signed_distance > 0 else -1

def detect_crossing(vehicle, center, config):
    """用接地點軌跡與有限停止線相交來判斷 IN／OUT。"""
    line_start, line_end = config["line"]
    current_side = get_line_side(center, line_start, line_end)
    if current_side == 0:
        return None
    if vehicle.line_side is None:
        vehicle.line_side = current_side
        vehicle.line_side_point = center
        return None
    if current_side == vehicle.line_side:
        vehicle.line_side_point = center
        return None

    previous_side = vehicle.line_side
    previous_point = vehicle.line_side_point or vehicle.last_center
    vehicle.line_side = current_side
    vehicle.line_side_point = center
    if segment_intersection(previous_point, center, line_start, line_end) is None:
        return None
    transition = (previous_side, current_side)
    return "IN" if transition == config["in_transition"] else "OUT"


def load_saved_line(default_line, frame_size):
    """讀取網頁上一次儲存的兩點；檔案不存在或格式錯誤時使用預設值。"""
    if not POINTS_FILE.exists():
        return default_line
    try:
        with POINTS_FILE.open(encoding="utf-8") as file:
            saved = json.load(file)
        if "stop_line" in saved:
            line = saved["stop_line"]["line"]
            points = (line["L"], line["R"])
        else:
            points = saved["points"]
        if len(points) != 2:
            raise ValueError("停止線必須剛好有兩個端點")
        width, height = frame_size
        normalized = saved.get("video", {}).get("coordinate_system") == "normalized_0_to_1"
        if normalized:
            return tuple((
                round(float(point["x"]) * (width - 1)),
                round(float(point["y"]) * (height - 1)),
            ) for point in points)
        return tuple((round(float(point["x"])), round(float(point["y"]))) for point in points)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"無法讀取 {POINTS_FILE}，改用預設計數線：{error}")
        return default_line


def prepare_configuration_frame(
    video_source,
    source_name,
    camera_backend,
    camera_width,
    camera_height,
    camera_fps,
):
    """讀取來源第一幀，讓使用者能先在網頁完成校正。"""
    cap = open_video_capture(
        video_source,
        camera_backend,
        camera_width,
        camera_height,
        camera_fps,
    )
    if not cap.isOpened():
        set_runtime_status("error", f"無法開啟影像來源：{source_name}")
        return
    ok, frame = cap.read()
    cap.release()
    if not ok:
        set_runtime_status("error", f"無法讀取影像來源第一幀：{source_name}")
        return
    height, width = frame.shape[:2]
    config = get_video_config(source_name, width, height)
    stop_line = load_saved_line(config["line"], (width, height))
    with line_lock:
        app.state.frame_size = (width, height)
        app.state.video_config = {**config, "line": stop_line}
        app.state.stop_line = [{"x": x, "y": y} for x, y in stop_line]
        app.state.gates = []
        app.state.road_boundaries = None
        app.state.visual_scale_status = None
        app.state.inference_devices = None
    frame_broadcaster.publish(frame)
    set_runtime_status("configuring", "請標示停止線與至少三個 Gate，再按開始判斷")


def set_runtime_status(state, message):
    with line_lock:
        runtime_status.update(state=state, message=message)


def resolve_device(requested_device):
    """依實際可用加速裝置選擇 YOLO 推論，否則使用 CPU。"""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested_device == "mps" and not torch.backends.mps.is_available():
        print("要求使用 MPS，但目前不可用；改用 CPU。")
        return "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("要求使用 CUDA，但目前不可用；改用 CPU。")
        return "cpu"
    return requested_device


def resolve_road_marking_device(selected_device):
    """Stage-5 避開可能不支援的 MPS 算子；CUDA 與 CPU 維持原選擇。"""
    return "cpu" if selected_device == "mps" else selected_device


def reload_road_markings(frame, checkpoint_path, device, max_side):
    """只在操作員要求時載入 Stage-5，對當前影格推論一次後釋放模型。"""
    detector = RoadMarkingDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        max_side=max_side,
    )
    try:
        return detector.predict(frame)
    finally:
        del detector
        if device == "cuda":
            torch.cuda.empty_cache()


def road_markings_for_frame(frame, frame_count, checkpoint_path, device, max_side):
    """只處理一次待執行的 Reload；其餘影格直接沿用上次結果。"""
    with line_lock:
        cached_result = getattr(app.state, "road_marking_result", None)
        reload_requested = getattr(app.state, "road_marking_reload_requested", False)
        if reload_requested:
            app.state.road_marking_reload_requested = False
    if not reload_requested or checkpoint_path is None:
        return cached_result

    try:
        new_result = reload_road_markings(frame, checkpoint_path, device, max_side)
    except Exception as error:
        message = f"Stage-5 重新偵測失敗：{type(error).__name__}: {error}"
        print(message)
        with line_lock:
            road_marking_status.update(state="error", message=message)
        return cached_result

    with line_lock:
        app.state.road_marking_result = new_result
        road_marking_status.update(
            enabled=True,
            state="ready",
            message=(
                f"Stage-5 已重新偵測（{device}，"
                f"{new_result.inference_ms:.0f} ms）；標線持續顯示"
            ),
            result=new_result.as_dict(frame_count),
        )
    return new_result


def run_monitor(
    graphic_device,
    model_path,
    road_model_path,
    road_marking_max_side,
    road_marking_opacity,
    video_source,
    source_name,
    is_live_source,
    camera_backend,
    camera_width,
    camera_height,
    camera_fps,
    vehicle_model_path=VEHICLE_MODEL_PATH,
    detection_mode=DETECTION_MODE_LEAVING,
):
    """開啟影片或攝影機，執行號誌辨識、車輛追蹤與道路標線辨識。"""
    if not is_live_source and START_TIME_SECONDS < 0:
        raise ValueError("START_TIME_SECONDS 不可小於 0。")
    if (
        not is_live_source
        and END_TIME_SECONDS is not None
        and END_TIME_SECONDS <= START_TIME_SECONDS
    ):
        raise ValueError("END_TIME_SECONDS 必須大於 START_TIME_SECONDS。")

    (
        primary_classes,
        primary_vehicle_classes,
        custom_vehicle_classes,
    ) = detection_plan(detection_mode)

    # 官方 YOLO11 資產名稱缺少時讓 Ultralytics 自動下載；
    # 其他模型路徑則必須是實際存在的檔案，避免把拼錯路徑當成模型名稱。
    is_official_model = model_path in OFFICIAL_YOLO_MODELS
    resolved_model_path = Path(model_path).expanduser().resolve()
    if not is_official_model and not resolved_model_path.is_file():
        message = f"找不到 YOLO 模型檔：{model_path}。請用 --model 指定正確的 .pt 檔。"
        print(message)
        set_runtime_status("error", message)
        frame_broadcaster.finish()
        return
    resolved_vehicle_model_path = Path(vehicle_model_path).expanduser().resolve()
    if not resolved_vehicle_model_path.is_file():
        message = f"找不到車輛 YOLO 模型檔：{vehicle_model_path}。"
        print(message)
        set_runtime_status("error", message)
        frame_broadcaster.finish()
        return

    selected_device = resolve_device(graphic_device)
    image_size = MPS_IMAGE_SIZE if selected_device == "mps" else CPU_IMAGE_SIZE
    model_description = f"{Path(model_path).name} + {resolved_vehicle_model_path.name}"
    set_runtime_status(
        "loading",
        f"正在載入 YOLO 模型：{model_description}（{selected_device}，{image_size}px）",
    )
    try:
        primary_model = YOLO(model_path if is_official_model else resolved_model_path)
        vehicle_model = YOLO(resolved_vehicle_model_path)
    except Exception as error:
        message = f"無法載入 YOLO 模型：{error}"
        print(message)
        set_runtime_status("error", message)
        frame_broadcaster.finish()
        return

    resolved_road_model_path = (
        Path(road_model_path).expanduser().resolve()
        if road_model_path is not None else None
    )
    road_device = resolve_road_marking_device(selected_device)
    with line_lock:
        app.state.inference_devices = {
            "yolo": selected_device,
            "stage5": road_device if resolved_road_model_path is not None else None,
        }
    cap = open_video_capture(
        video_source,
        camera_backend,
        camera_width,
        camera_height,
        camera_fps,
    )
    if not cap.isOpened():
        message = f"無法開啟影像來源：{source_name}"
        print(message)
        set_runtime_status("error", message)
        frame_broadcaster.finish()
        return

    ret, frame = cap.read()
    if not ret:
        message = f"無法讀取影像來源第一張影格：{source_name}"
        print(message)
        cap.release()
        set_runtime_status("error", message)
        frame_broadcaster.finish()
        return

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        if is_live_source:
            fps = float(camera_fps or 30.0)
            print(f"攝影機未回報 FPS，使用 {fps:.1f} FPS 作為時間基準。")
        else:
            cap.release()
            raise RuntimeError("無法取得影片 FPS，因此無法套用開始／結束時間設定。")

    if is_live_source:
        total_frames = 0
        video_duration = None
        start_frame = 0
        end_frame = 0
        pending_frame = frame
    else:
        pending_frame = None
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

    print(f"Input source: {source_name}")
    print(f"Video resolution: {width}x{height} @ {fps:.1f} FPS")
    if is_live_source:
        print("Processing interval: live stream")
    else:
        end_label = f"{end_frame / fps:.2f}" if end_frame > 0 else "video end"
        print(f"Processing interval: {start_frame / fps:.2f}s - {end_label}s")

    with line_lock:
        saved_config = app.state.video_config
        video_config = dict(saved_config)
        video_config["line"] = tuple(
            (round(point["x"] * (width - 1)), round(point["y"] * (height - 1)))
            for point in app.state.stop_line
        )
        speed_gates = [dict(gate) for gate in app.state.gates]
        speed_boundaries = getattr(app.state, "road_boundaries", None)
    line_start, line_end = video_config["line"]
    print(f"Counting line for {source_name}: start={line_start}, end={line_end}")
    light_detector = TrafficLightDetector()
    speed_estimator = GateSpeedEstimator(
        speed_gates, (width, height), speed_boundaries,
        stop_line=video_config["line"],
        known_gate_spacing=True,
    )
    with line_lock:
        app.state.visual_scale_status = speed_estimator.metric_calibrator.as_dict()

    vehicles = {}
    fallback_tracker = FallbackVehicleTracker()
    total_vehicle_ids = set()
    illegal_vehicle_ids = set()
    direction_counts = {"IN": 0, "OUT": 0}
    # frame_count 保留原始影片中的影格編號，方便回查違規發生位置。
    frame_count = start_frame
    traffic_light_state = "UNKNOWN"
    traffic_light_color = TrafficLightDetector.COLORS["UNKNOWN"]
    red_ratio = 0.0
    green_ratio = 0.0
    previous_light_state = "UNKNOWN"
    previous_time = perf_counter()
    last_video_timestamp = None
    smoothed_fps = 0.0
    previous_gray = None
    camera_motion = 0.0
    camera_motion_inlier_ratio = 0.0
    with line_lock:
        latest_road_result = getattr(app.state, "road_marking_result", None)
        if (
            latest_road_result is not None
            and latest_road_result.class_map.shape != (height, width)
        ):
            latest_road_result = None
            app.state.road_marking_result = None
            road_marking_status.update(
                state="idle",
                message="畫面尺寸已變更，請按 Reload LaneNet 重新偵測標線",
                result=None,
            )

    

    set_runtime_status("running", "判斷執行中")
    print("Running web monitor. Open http://127.0.0.1:8000")

    while cap.isOpened():
        if getattr(app.state, "cancel_requested", False):
            break
        # 結束時間採不包含上限的區間，例如 5~10 秒不會處理第 10 秒的影格。
        if (
            not is_live_source
            and end_frame > 0
            and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) >= end_frame
        ):
            break
        if pending_frame is not None:
            ret, frame = True, pending_frame
            pending_frame = None
        else:
            ret, frame = cap.read()
        if not ret:
            if is_live_source:
                print(f"攝影機畫面中斷，停止處理：{source_name}")
            break
        frame_count += 1
        if is_live_source:
            frame_timestamp = perf_counter()
        else:
            capture_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            fallback_timestamp = frame_count / fps
            frame_timestamp = (
                capture_timestamp
                if math.isfinite(capture_timestamp)
                and capture_timestamp > 0
                and (last_video_timestamp is None or capture_timestamp > last_video_timestamp)
                else fallback_timestamp
            )
            if last_video_timestamp is not None and frame_timestamp <= last_video_timestamp:
                frame_timestamp = last_video_timestamp + 1.0 / fps
            last_video_timestamp = frame_timestamp

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (
            camera_transform,
            camera_motion,
            camera_motion_inlier_ratio,
        ) = estimate_camera_motion(previous_gray, current_gray)
        previous_gray = current_gray

        primary_track_options = {
            "persist": True,
            "tracker": "bytetrack.yaml",
            "classes": primary_classes,
            "verbose": False,
        }
        inference_frame = frame
        primary_result = primary_model.track(
            inference_frame,
            **primary_track_options,
            device=selected_device,
            imgsz=image_size,
        )[0]
        vehicle_result = vehicle_model.track(
            inference_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=sorted(custom_vehicle_classes),
            verbose=False,
            device=selected_device,
            imgsz=image_size,
        )[0]

        latest_road_result = road_markings_for_frame(
            inference_frame,
            frame_count,
            resolved_road_model_path,
            road_device,
            road_marking_max_side,
        )

        if latest_road_result is not None:
            frame = draw_road_markings(
                frame,
                latest_road_result,
                opacity=road_marking_opacity,
            )

        for gate in speed_estimator.gates:
            left = tuple(round(value) for value in gate.left)
            right = tuple(round(value) for value in gate.right)
            cv2.line(frame, left, right, (255, 220, 0), 1)
            cv2.putText(
                frame, gate.name, (left[0] + 3, max(left[1] - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 0), 1,
            )

        boxes = primary_result.boxes.xyxy.cpu().numpy().astype(int)
        classes = primary_result.boxes.cls.cpu().numpy().astype(int)
        confidences = primary_result.boxes.conf.cpu().numpy()
        draw_pedestrians(frame, boxes, classes, confidences)
        light_box = choose_traffic_light(boxes, classes, confidences)
        if light_box is not None:
            roi, (x1, y1, x2, y2) = safe_roi(inference_frame, light_box)
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

        uses_two_vehicle_sources = bool(
            primary_vehicle_classes and custom_vehicle_classes
        )
        primary_vehicle_detections = tracked_vehicle_detections(
            primary_result,
            primary_vehicle_classes,
            track_namespace=0 if uses_two_vehicle_sources else None,
        )
        custom_vehicle_detections = tracked_vehicle_detections(
            vehicle_result,
            custom_vehicle_classes,
            track_namespace=1 if uses_two_vehicle_sources else None,
        )
        vehicle_detections = merge_vehicle_detections(
            primary_vehicle_detections, custom_vehicle_detections,
        )
        vehicle_boxes = [detection[0] for detection in vehicle_detections]
        vehicle_ids = [detection[1] for detection in vehicle_detections]
        vehicle_polygons = [detection[2] for detection in vehicle_detections]
        missing_indices = [
            index for index, track_id in enumerate(vehicle_ids) if track_id < 0
        ]
        temporary_ids = fallback_tracker.assign(
            [vehicle_boxes[index] for index in missing_indices], frame_count
        )
        for index, temporary_id in zip(missing_indices, temporary_ids):
            vehicle_ids[index] = temporary_id
        active_formal_ids = [track_id for track_id in vehicle_ids if track_id >= 0]
        for formal_id, temporary_id in fallback_tracker.claim_temporary_ids(
            vehicle_boxes, vehicle_ids, frame_count, active_formal_ids,
            speed_estimator.tracks.keys(),
        ):
            speed_estimator.reassign_track(formal_id, temporary_id)
        for temporary_id, formal_id in fallback_tracker.claim_formal_ids(
            vehicle_boxes, vehicle_ids, frame_count, temporary_ids,
            speed_estimator.tracks.keys(),
        ):
            speed_estimator.reassign_track(temporary_id, formal_id)
        fallback_tracker.observe_formal_ids(vehicle_boxes, vehicle_ids, frame_count)

        for box, track_id, polygon in zip(
            vehicle_boxes, vehicle_ids, vehicle_polygons
        ):
            x1, y1, x2, y2 = box
            center = vehicle_ground_contact(box, polygon)
            has_tracker_id = track_id >= 0
            if has_tracker_id:
                total_vehicle_ids.add(track_id)
            if track_id not in vehicles:
                vehicles[track_id] = Vehicle(
                    track_id=track_id,
                    last_center=center,
                    first_seen_frame=frame_count,
                    last_seen_frame=frame_count,
                )
            vehicle = vehicles[track_id]

            if has_tracker_id:
                direction = detect_crossing(vehicle, center, video_config)
                if direction == "IN" and vehicle.crossed_in:
                    direction = None
                elif direction == "OUT" and vehicle.crossed_out:
                    direction = None
                if direction:
                    is_new_violation = vehicle.mark_crossing(
                        direction, traffic_light_state
                    )
                    direction_counts[direction] += 1
                    print(
                        f"Frame {frame_count}: vehicle {track_id} crossed "
                        f"{direction}, light={traffic_light_state}"
                    )
                    if is_new_violation:
                        illegal_vehicle_ids.add(track_id)
                        print(f"  -> violation: vehicle {track_id}")

            vehicle.update_position(
                center,
                frame_count,
                MOTION_STOP_THRESHOLD,
                MOTION_MOVING_THRESHOLD,
                MOTION_CONFIRM_FRAMES,
                MOTION_SMOOTHING_ALPHA,
                camera_transform,
            )
            speed_estimate = speed_estimator.update(
                track_id, center, frame_timestamp, frame_count
            )
            speed_zone_state = speed_estimator.speed_zone_state(track_id)
            section_estimate = speed_estimator.section_speed(track_id)
            color = (
                (0, 0, 255) if vehicle.illegal else
                (255, 0, 0) if vehicle.motion_state == "STOPPED" else
                (0, 255, 0)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            has_speed_label = speed_zone_state in ("inside", "after")
            speed_y = max(y1 - 10, 36 if has_speed_label else 18)
            label_y = speed_y - 18 if has_speed_label else speed_y
            if has_tracker_id:
                label = "ILLEGAL" if vehicle.illegal else "OK"
                identity = f"ID:{track_id} {label}"
            else:
                identity = f"TEMP:{-track_id}"
            cv2.putText(
                frame,
                f"{identity} {vehicle.motion_state} {vehicle.cross_direction or ''}",
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
            if speed_zone_state == "inside":
                if speed_estimate is None:
                    speed_label = "EST -- km/h (estimate)"
                else:
                    qualifier = (
                        " (estimate)"
                        if speed_estimate.confidence == "PROVISIONAL" else ""
                    )
                    speed_label = (
                        f"EST {speed_estimate.kmh:.1f} km/h{qualifier} "
                        f"L{speed_estimate.lane_number}"
                    )
                speed_width = cv2.getTextSize(
                    speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )[0][0]
                speed_x = max(0, min(x1, frame.shape[1] - speed_width - 4))
                cv2.putText(
                    frame, speed_label, (speed_x, speed_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )
            elif speed_zone_state == "after":
                if section_estimate is None:
                    section_label = "SECTION -- km/h"
                else:
                    pair = (
                        f"{section_estimate.from_gate.replace('Gate ', 'G')}-"
                        f"{section_estimate.to_gate.replace('Gate ', 'G')}"
                    )
                    section_label = f"SECTION {section_estimate.kmh:.1f} km/h {pair}"
                section_width = cv2.getTextSize(
                    section_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )[0][0]
                section_x = max(0, min(x1, frame.shape[1] - section_width - 4))
                cv2.putText(
                    frame, section_label, (section_x, speed_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                )

        # 只移除確實消失一段時間的追蹤目標，避免仍在畫面中的車輛遺失狀態。
        stale_ids = [
            track_id
            for track_id, vehicle in vehicles.items()
            if frame_count - vehicle.last_seen_frame > (
                5 if track_id < 0 else TRACK_TTL_FRAMES
            )
        ]
        for track_id in stale_ids:
            del vehicles[track_id]
            speed_estimator.remove(track_id)

        now = perf_counter()
        instant_fps = 1.0 / max(now - previous_time, 1e-9)
        smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
        previous_time = now

        # 直接畫出目前網頁設定的計數線；端點順序也決定 IN／OUT 的側別。
        line_start, line_end = video_config["line"]
        cv2.line(frame, line_start, line_end, (0, 0, 255), 1)
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
            if latest_road_result is not None:
                cv2.putText(
                    frame,
                    (
                        f"Stage-5: {latest_road_result.inference_ms:.0f}ms | "
                        f"markings: {len(latest_road_result.objects)} | cached"
                    ),
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        frame_broadcaster.publish(frame)

    cap.release()
    with line_lock:
        app.state.processing_started = False
        if getattr(app.state, "road_marking_reload_requested", False):
            app.state.road_marking_reload_requested = False
            road_marking_status.update(
                state="error",
                message="影片已結束，Reload LaneNet 未執行",
            )

    if getattr(app.state, "cancel_requested", False):
        print("收到重置訊號，影片處理已中斷。")
    frame_broadcaster.finish()
    finished_message = "攝影機串流已停止" if is_live_source else "影片處理完成"
    set_runtime_status("finished", finished_message)

    print("\n========== Summary ==========")
    print(f"Unique tracked vehicles: {len(total_vehicle_ids)}")
    print(f"Violating vehicles: {len(illegal_vehicle_ids)}")
    print(f"Crossing events: {sum(direction_counts.values())}")
    print(f"IN: {direction_counts['IN']}")
    print(f"OUT: {direction_counts['OUT']}")
    print("=============================")


app = FastAPI(title="Traffic Monitoring System")


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(PROJECT_ROOT / "frontend" / "main.html")


@app.get("/video_feed")
def video_feed():
    """提供已完成辨識與繪圖的 MJPEG 影像串流。"""
    return StreamingResponse(
        frame_broadcaster.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/first_frame.jpg")
def first_frame():
    """校正用的單張第一幀；避免瀏覽器等待 MJPEG 後續影格而顯示黑畫面。"""
    frame = frame_broadcaster.latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="第一幀尚未準備完成")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/api/status")
def status():
    with line_lock:
        return {
            **runtime_status,
            "processing_started": getattr(app.state, "processing_started", False),
            "road_markings": {
                key: value
                for key, value in road_marking_status.items()
                if key != "result"
            },
            "stop_line": getattr(app.state, "stop_line", None),
            "gates": getattr(app.state, "gates", []),
            "road_boundaries": getattr(app.state, "road_boundaries", None),
            "visual_scale": getattr(app.state, "visual_scale_status", None),
            "speed_scale": getattr(app.state, "visual_scale_status", None),
            "devices": getattr(app.state, "inference_devices", None),
            "detection_mode": getattr(
                app.state, "detection_mode", DETECTION_MODE_LEAVING
            ),
            "frame_size": getattr(app.state, "frame_size", None),
        }


@app.get("/api/road_markings")
def road_markings():
    """回傳最近一次 Reload 的固定標線；同時含像素與 0~1 正規化座標。"""
    with line_lock:
        return dict(road_marking_status)


@app.post("/api/reload_lanenet")
def request_lanenet_reload():
    """排程在下一張影格載入 Stage-5 並重新偵測一次。"""
    with line_lock:
        if getattr(app.state, "road_model_path", None) is None:
            raise HTTPException(status_code=409, detail="Stage-5 已停用")
        if not getattr(app.state, "processing_started", False):
            raise HTTPException(status_code=409, detail="請先開始判斷")
        if road_marking_status["state"] == "loading":
            raise HTTPException(status_code=409, detail="Stage-5 正在重新偵測")
        app.state.road_marking_reload_requested = True
        road_marking_status.update(
            state="loading",
            message="等待下一張影格，準備重新載入 Stage-5",
        )
    return {"status": "queued", "message": "Stage-5 重新偵測已排程"}


@app.post("/save_points")
async def save_points(request: Request):
    """儲存以影片寬高比例表示的停止線與多個 Gate 校正資料。"""
    try:
        data = await request.json()
        detection_mode = data.get("detection_mode", DETECTION_MODE_LEAVING)
        if detection_mode not in DETECTION_MODES:
            raise ValueError("主要偵測模式必須是 entering 或 leaving")
        stop_line = data["stop_line"]
        gates = data["gates"]
        if not isinstance(gates, list) or len(gates) < 3:
            raise ValueError("至少需要三個 Gate")
        width, height = app.state.frame_size

        def point_of(item):
            x, y = float(item["x"]), float(item["y"])
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("座標比例必須介於 0 和 1")
            return {"x": round(x, 5), "y": round(y, 5)}

        def line_of(item):
            line = item["line"]
            points = ((float(line["L"]["x"]), float(line["L"]["y"])), (float(line["R"]["x"]), float(line["R"]["y"])))
            # 新版前端傳 0~1 比例；為相容舊網頁快取，像素座標也轉為比例。
            if any(value > 1 for point in points for value in point):
                points = tuple((x / max(width - 1, 1), y / max(height - 1, 1)) for x, y in points)
            normalized = tuple(point_of({"x": x, "y": y}) for x, y in points)
            if math.hypot(
                (normalized[1]["x"] - normalized[0]["x"]) * (width - 1),
                (normalized[1]["y"] - normalized[0]["y"]) * (height - 1),
            ) < 2:
                raise ValueError("線段長度不可小於 2 像素")
            return normalized

        stop_points = line_of(stop_line)
        normalized_gates = []
        for gate in gates:
            line = line_of(gate)
            normalized_gate = {
                "gate": str(gate["gate"]),
                "lane": int(gate["lane"]),
                "station_m": float(gate["station_m"]),
                "line": {"L": line[0], "R": line[1]},
            }
            if "anchor" in gate or "angle_deg" in gate:
                normalized_gate["anchor"] = point_of(gate["anchor"])
                normalized_gate["angle_deg"] = float(gate["angle_deg"])
                if not math.isfinite(normalized_gate["angle_deg"]):
                    raise ValueError("Gate 角度須為有限數字")
            normalized_gates.append(normalized_gate)
        if any(gate["lane"] < 1 for gate in normalized_gates):
            raise ValueError("lane（車道數）至少為 1")
        if len({gate["gate"] for gate in normalized_gates}) != len(normalized_gates):
            raise ValueError("Gate 名稱不可重複")
        if any(not math.isfinite(gate["station_m"]) for gate in normalized_gates):
            raise ValueError("Gate 的 station_m 須為有限數字")
        if any(b["station_m"] <= a["station_m"] for a, b in zip(normalized_gates, normalized_gates[1:])):
            raise ValueError("Gate 的 station_m 必須依 Gate 順序遞增")
        for gate, station in zip(normalized_gates, known_gate_stations(normalized_gates)):
            gate["station_m"] = station
        road_boundaries = None
        if "road_boundaries" in data:
            raw_boundaries = data["road_boundaries"]
            road_boundaries = {}
            for side in ("L", "R"):
                raw_points = raw_boundaries[side]
                if not isinstance(raw_points, list) or len(raw_points) < 2:
                    raise ValueError(f"{side} 道路邊界至少需要兩個點")
                road_boundaries[side] = [point_of(point) for point in raw_points]
                start = road_boundaries[side][0]
                stop = stop_points[0 if side == "L" else 1]
                if math.hypot(
                    (start["x"] - stop["x"]) * (width - 1),
                    (start["y"] - stop["y"]) * (height - 1),
                ) > 2:
                    raise ValueError(f"{side} 道路邊界必須從停止線端點開始")
            lane_count = int(data["lane_count"])
            if lane_count < 1 or any(gate["lane"] != lane_count for gate in normalized_gates):
                raise ValueError("所有 Gate 的車道數必須與道路車道數一致")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail=f"無效的計數線：{error}") from error

    with line_lock:
        app.state.video_config["line"] = tuple((point["x"], point["y"]) for point in stop_points)
        app.state.stop_line = list(stop_points)
        app.state.gates = normalized_gates
        app.state.road_boundaries = road_boundaries
        app.state.detection_mode = detection_mode

    POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3 if road_boundaries is not None else 2,
        "video": {
            "name": getattr(app.state, "source_name", DEFAULT_VIDEO_NAME),
            "coordinate_system": "normalized_0_to_1",
            "width": width,
            "height": height,
        },
        "stop_line": {"line": {"L": stop_points[0], "R": stop_points[1]}},
        "detection_mode": detection_mode,
        "gates": normalized_gates,
    }
    if road_boundaries is not None:
        payload["lane_count"] = lane_count
        payload["road_boundaries"] = road_boundaries
    with POINTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"Gate 校正資料已寫入：{POINTS_FILE}")
    return {"status": "success", "message": "設定已儲存", "gates": normalized_gates}


@app.post("/start_processing")
def start_processing():
    with line_lock:
        if len(getattr(app.state, "gates", [])) < 3:
            raise HTTPException(status_code=422, detail="請先儲存至少三個 Gate")
        if getattr(app.state, "processing_started", False):
            raise HTTPException(status_code=409, detail="判斷已經開始")
        app.state.processing_started = True
        app.state.cancel_requested = False
        frame_broadcaster.reset()
        thread = Thread(target=app.state.monitor_worker, daemon=True)
        app.state.monitor_thread = thread
        thread.start()
    return {"status": "started"}

@app.post("/api/reset")
def reset_processing():
    with line_lock:
        thread = getattr(app.state, "monitor_thread", None)
        if getattr(app.state, "processing_started", False):
            app.state.cancel_requested = True
    # 等待整個工作執行緒（包含 finally）結束，才讓新一輪判斷開始。
    if thread is not None and thread.is_alive():
        thread.join()
    with line_lock:
        app.state.cancel_requested = False
        app.state.monitor_thread = None
        app.state.visual_scale_status = None
        if getattr(app.state, "road_marking_reload_requested", False):
            app.state.road_marking_reload_requested = False
            road_marking_status.update(
                state="error",
                message="Reload LaneNet 已取消；保留上次標線",
            )
    frame_broadcaster.reset()

    return {"status": "success", "message": "後端已重置"}


def main():
    parser = argparse.ArgumentParser(description="交通監控網頁串流服務")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        help="影片、串流網址或影像來源路徑；未指定時依 DEFAULT_INPUT_MODE 選擇來源",
    )
    source_group.add_argument(
        "--camera",
        type=int,
        help="實體／虛擬攝影機索引，例如 OBS Virtual Camera 常見為 0 或 1",
    )
    parser.add_argument(
        "--camera-backend",
        choices=tuple(CAMERA_BACKENDS),
        default="auto",
        help="預設 auto；若特定攝影機不相容，可改用 dshow 或 msmf",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help=f"要求的攝影機寬度（預設 {DEFAULT_CAMERA_WIDTH}）",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help=f"要求的攝影機高度（預設 {DEFAULT_CAMERA_HEIGHT}）",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help=f"要求的攝影機 FPS（預設 {DEFAULT_CAMERA_FPS:g}）",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="探測攝影機索引 0～9 後結束",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="預設 auto：依序使用 CUDA、MPS、CPU；指定的加速裝置不可用時使用 CPU",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="預設 YOLO 權重（預設 yolo11x.pt）",
    )
    parser.add_argument(
        "--vehicle-model",
        default=str(VEHICLE_MODEL_PATH),
        help="scooter 及離鏡頭汽車 YOLO 權重（預設 models/my_model.pt）",
    )
    parser.add_argument(
        "--road-model",
        default=str(ROAD_MARKING_MODEL_PATH),
        help="Stage-5 道路標線 checkpoint 路徑",
    )
    parser.add_argument(
        "--no-road-markings",
        action="store_true",
        help="只執行原本的 YOLO，不載入 Stage-5",
    )
    parser.add_argument(
        "--road-marking-max-side",
        type=int,
        default=1280,
        help="Stage-5 推論影像長邊；越大越準但越慢",
    )
    parser.add_argument(
        "--road-marking-opacity",
        type=float,
        default=0.38,
        help="道路標線遮罩透明度（0~1）",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    if args.list_cameras:
        cameras = list_available_cameras(args.camera_backend)
        if not cameras:
            print("找不到可讀取的攝影機。請先啟動虛擬攝影機，再改用其他 backend 重試。")
        else:
            print("可使用的攝影機索引：")
            for index, width, height, fps in cameras:
                print(f"  {index}: {width}x{height}, FPS={fps:.1f}")
        return
    if args.source is None and args.camera is None:
        if DEFAULT_INPUT_MODE == "camera":
            args.camera = DEFAULT_CAMERA_INDEX
            print(f"未指定影像來源，預設使用虛擬相機索引 {args.camera}")
        elif DEFAULT_INPUT_MODE == "video":
            args.source = str(PROJECT_ROOT / "inputVideos" / DEFAULT_VIDEO_NAME)
            print(f"未指定影像來源，預設使用影片 {args.source}")
        else:
            parser.error('DEFAULT_INPUT_MODE 必須是 "camera" 或 "video"')
    if args.camera is not None and args.camera < 0:
        parser.error("--camera 不可小於 0")
    for option_name in ("camera_width", "camera_height", "camera_fps"):
        option_value = getattr(args, option_name)
        if option_value is not None and option_value <= 0:
            parser.error(f"--{option_name.replace('_', '-')} 必須大於 0")
    if not 0.0 <= args.road_marking_opacity <= 1.0:
        parser.error("--road-marking-opacity 必須介於 0 和 1")

    if args.camera is not None:
        video_source = args.camera
        source_name = f"camera:{args.camera}"
        is_live_source = True
    else:
        video_source = args.source
        source_name = Path(args.source).name or args.source
        is_live_source = args.source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )

    app.state.source_name = source_name
    app.state.road_model_path = None if args.no_road_markings else args.road_model
    app.state.road_marking_reload_requested = False
    app.state.road_marking_result = None
    app.state.visual_scale_status = None
    app.state.detection_mode = DETECTION_MODE_LEAVING
    with line_lock:
        road_marking_status.update(
            enabled=not args.no_road_markings,
            state="disabled" if args.no_road_markings else "idle",
            message=(
                "Stage-5 已停用" if args.no_road_markings
                else "按 Reload LaneNet 偵測一次道路標線"
            ),
            result=None,
        )

    def monitor_worker():
        try:
            with line_lock:
                detection_mode = getattr(
                    app.state, "detection_mode", DETECTION_MODE_LEAVING
                )
            run_monitor(
                args.device,
                args.model,
                None if args.no_road_markings else args.road_model,
                args.road_marking_max_side,
                args.road_marking_opacity,
                video_source,
                source_name,
                is_live_source,
                args.camera_backend,
                args.camera_width,
                args.camera_height,
                args.camera_fps,
                args.vehicle_model,
                detection_mode,
            )
        except Exception as error:
            message = f"判斷程式意外停止：{type(error).__name__}: {error}"
            print(message)
            set_runtime_status("error", message)
            frame_broadcaster.finish()
        finally:
            with line_lock:
                app.state.processing_started = False
                app.state.road_marking_reload_requested = False
                if road_marking_status["state"] == "loading":
                    road_marking_status.update(
                        state="error",
                        message="影片處理已停止，Reload LaneNet 未完成",
                    )

    app.state.monitor_worker = monitor_worker
    app.state.processing_started = False
    prepare_configuration_frame(
        video_source,
        source_name,
        args.camera_backend,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        frame_broadcaster.finish()


@app.post("/shutdown")
async def shutdown_server():
    """接收到請求後，關閉背景的 Python 伺服器"""
    # 發送 SIGINT 訊號給自己，等同於 Ctrl+C
    os.kill(os.getpid(), signal.SIGINT)
    return {"message": "Server is shutting down..."}

if __name__ == "__main__":
    main()
