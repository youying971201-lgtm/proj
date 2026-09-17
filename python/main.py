import argparse
import json
import math
import os
from pathlib import Path
from threading import Condition, Lock, Thread
from time import perf_counter, sleep

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
from gate_speed import GateSpeedEstimator
from road_marking import RoadMarkingDetector, draw_road_markings
from trafficLight import TrafficLightDetector
from video_source import CAMERA_BACKENDS, list_available_cameras, open_video_capture

import os
import signal
import threading
import time


# 基本設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 影像來源預設值：改成 "camera" 使用攝影機，改成 "video" 使用下方影片。
DEFAULT_INPUT_MODE = "video"
DEFAULT_VIDEO_NAME = "IMG_6898.MOV"
DEFAULT_CAMERA_INDEX = 1
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30.0
# 預設使用 COCO 預訓練的 YOLO11s。第一次啟動會由 Ultralytics 自動下載。
MODEL_PATH = "yolo11x.pt"
ROAD_MARKING_MODEL_PATH = (
    PROJECT_ROOT / "models" / "road_marking" / "stage5_roadmark_best.pth"
)
START_TIME_SECONDS = 0.0 #影片開始時間
END_TIME_SECONDS = None # 影片結束時間
DEVICE = None #顯示裝置
# YOLO11 COCO 類別：car=2、motorcycle=3、bus=5、truck=7、traffic light=9。
CLASSES_TO_DETECT = [2, 3, 5, 7, 9]
VEHICLE_CLASSES = {2, 3, 5, 7}
MPS_IMAGE_SIZE = 640  # M2 Air GPU 上兼顧偵測品質與長時間運行溫度／速度。
CPU_IMAGE_SIZE = 512  # MPS 暫時不可用時，避免 YOLO11s 在 CPU 上過度延遲。
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

    def latest_frame(self):
        with self._condition:
            return self._frame


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
        line_y = round(height * 0.7)
        return {
            "line": ((round(width * 0.15), line_y), (round(width * 0.85), line_y)),
            "in_transition": (-1, 1),
            "playback_speed": 1.0,
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
    frame_broadcaster.publish(frame)
    set_runtime_status("configuring", "請標示停止線與至少三個 Gate，再按開始判斷")


def set_runtime_status(state, message):
    with line_lock:
        runtime_status.update(state=state, message=message)


def resolve_device(requested_device):
    """在 Apple Silicon 優先採用 MPS，無法使用時安全回退到 CPU。"""
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        print("MPS 目前不可用，改用 CPU。可用 `python -c` 檢查 torch.backends.mps.is_available()。")
        return "cpu"
    if requested_device == "mps" and not torch.backends.mps.is_available():
        print("要求使用 MPS，但目前不可用；改用 CPU。")
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

    # yolo11s.pt 是 Ultralytics 官方資產名稱，缺少時讓套件自動下載；
    # 其他模型路徑則必須是實際存在的檔案，避免把拼錯路徑當成模型名稱。
    is_default_model = model_path == MODEL_PATH
    resolved_model_path = Path(model_path).expanduser().resolve()
    if not is_default_model and not resolved_model_path.is_file():
        message = f"找不到 YOLO 模型檔：{model_path}。請用 --model 指定正確的 .pt 檔。"
        print(message)
        set_runtime_status("error", message)
        frame_broadcaster.finish()
        return

    selected_device = resolve_device(graphic_device)
    image_size = MPS_IMAGE_SIZE if selected_device == "mps" else CPU_IMAGE_SIZE
    set_runtime_status(
        "loading",
        f"正在載入 YOLO 模型：{Path(model_path).name}（{selected_device}，{image_size}px）",
    )
    try:
        model = YOLO(model_path if is_default_model else resolved_model_path)
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
    else:
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
    playback_speed = video_config["playback_speed"]
    print(f"Counting line for {source_name}: start={line_start}, end={line_end}")
    print(f"Playback speed: {playback_speed:.0%}")
    light_detector = TrafficLightDetector()
    speed_estimator = GateSpeedEstimator(speed_gates, (width, height), speed_boundaries)

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
        ret, frame = cap.read()
        if not ret:
            if is_live_source:
                print(f"攝影機畫面中斷，準備重新連線：{source_name}")
                cap.release()
                continue
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

        track_options = {
            "persist": True,
            "tracker": "bytetrack.yaml",
            "classes": CLASSES_TO_DETECT,
            "verbose": False,
        }
        if DEVICE is not None:
            track_options["device"] = DEVICE
        inference_frame = frame
        result = model.track(
            inference_frame,
            **track_options,
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

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

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
                    center = ((x1 + x2) // 2, y2)
                    with line_lock:
                        active_config = dict(video_config)
                    direction = detect_crossing(vehicle, center, active_config)
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
                    speed_estimate = speed_estimator.update(
                        int(track_id), center, frame_timestamp, frame_count
                    )
                    if vehicle.motion_state == "STOPPED":
                        color = (255,0,0)
                    else:
                        color = (0,255,0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    cross_info = vehicle.cross_direction or ""
                    speed_y = max(y1 - 10, 22)
                    label_y = max(speed_y - 20, 18) if speed_estimate else speed_y
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {label} {vehicle.motion_state} {cross_info}",
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
                    if speed_estimate is not None:
                        cv2.putText(
                            frame,
                            f"EST {speed_estimate.kmh:.1f} km/h (Lane {speed_estimate.lane_number})",
                            (x1, speed_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )

        # 只移除確實消失一段時間的追蹤目標，避免仍在畫面中的車輛遺失狀態。
        stale_ids = [
            track_id
            for track_id, vehicle in vehicles.items()
            if frame_count - vehicle.last_seen_frame > TRACK_TTL_FRAMES
        ]
        for track_id in stale_ids:
            del vehicles[track_id]
            speed_estimator.remove(track_id)

        now = perf_counter()
        instant_fps = 1.0 / max(now - previous_time, 1e-9)
        smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
        previous_time = now

        # 直接畫出目前網頁設定的計數線；端點順序也決定 IN／OUT 的側別。
        with line_lock:
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
    print(f"Crossing events: {len(crossing_events)}")
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


def video_config_for_response():
    """回傳目前線段；背景工作尚未讀到影片大小前可能沒有設定。"""
    return getattr(app.state, "line", None)


@app.post("/save_points")
async def save_points(request: Request):
    """儲存以影片寬高比例表示的停止線與多個 Gate 校正資料。"""
    try:
        data = await request.json()
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
    if len(getattr(app.state, "gates", [])) < 3:
        raise HTTPException(status_code=422, detail="請先儲存至少三個 Gate")
    if getattr(app.state, "processing_started", False):
        raise HTTPException(status_code=409, detail="判斷已經開始")
    app.state.processing_started = True
    Thread(target=app.state.monitor_worker, daemon=True).start()
    return {"status": "started"}

@app.post("/api/reset")
def reset_processing():
    if getattr(app.state, "processing_started", False):
        app.state.cancel_requested = True
        # 等待背景迴圈確實中斷並把 processing_started 改為 False
        while getattr(app.state, "processing_started", False):
            time.sleep(0.1)
    
    # 狀態重置，準備下一次啟動
    app.state.cancel_requested = False
    with line_lock:
        if getattr(app.state, "road_marking_reload_requested", False):
            app.state.road_marking_reload_requested = False
            road_marking_status.update(
                state="error",
                message="Reload LaneNet 已取消；保留上次標線",
            )
    frame_broadcaster._finished = False
    with frame_broadcaster._condition:
        frame_broadcaster._sequence += 1
        frame_broadcaster._condition.notify_all()
    
    return {"status": "success", "message": "後端已重置"}


def main():
    parser = argparse.ArgumentParser(description="交通監控網頁串流服務")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        help="影片、串流網址或影像來源路徑；未指定時預設使用虛擬相機",
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
        help="預設 auto：依序使用 MPS、CUDA、CPU",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="YOLO 權重 .pt 檔路徑（預設 yolo11s.pt，首次會自動下載）",
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
