from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


# RLMD 的 25 類；目前只輸出使用者指定的 1~10 與 16。
ROAD_MARKING_CLASSES = {
    0: ("background", (0, 0, 0)),
    1: ("box junction", (0, 242, 255)),
    2: ("crosswalk", (76, 117, 34)),
    3: ("stop line", (204, 72, 61)),
    4: ("solid single white", (36, 28, 237)),
    5: ("solid single yellow", (164, 73, 163)),
    6: ("solid single red", (87, 122, 185)),
    7: ("solid double white", (21, 0, 136)),
    8: ("solid double yellow", (190, 146, 112)),
    9: ("dashed single white", (29, 230, 181)),
    10: ("dashed single yellow", (234, 217, 153)),
    11: ("left arrow", (76, 159, 158)),
    12: ("straight arrow", (134, 138, 121)),
    13: ("right arrow", (96, 64, 41)),
    14: ("left straight arrow", (146, 102, 7)),
    15: ("right straight arrow", (255, 153, 247)),
    16: ("channelizing line", (153, 204, 255)),
    17: ("motor prohibited", (153, 255, 155)),
    18: ("slow", (173, 153, 255)),
    19: ("motor priority lane", (147, 224, 230)),
    20: ("motor waiting zone", (87, 27, 35)),
    21: ("left turn box", (155, 158, 193)),
    22: ("motor icon", (78, 29, 109)),
    23: ("bike icon", (204, 164, 3)),
    24: ("parking lot", (185, 157, 175)),
}

DEFAULT_CLASS_IDS = tuple(range(1, 11)) + (16,)
AREA_CLASS_IDS = {1, 2, 3}
DASHED_CLASS_IDS = {9, 10}
LINE_CLASS_IDS = {4, 5, 6, 7, 8, 9, 10, 16}


@dataclass
class RoadMarkingResult:
    class_map: np.ndarray
    objects: List[Dict[str, Any]]
    inference_ms: float
    model_input_size: Tuple[int, int]

    def as_dict(self, frame_index: Optional[int] = None) -> Dict[str, Any]:
        result = {
            "inference_ms": round(self.inference_ms, 2),
            "model_input_size": {
                "width": self.model_input_size[0],
                "height": self.model_input_size[1],
            },
            "objects": self.objects,
        }
        if frame_index is not None:
            result["frame_index"] = frame_index
        return result


def _json_point(x: float, y: float, width: int, height: int) -> Dict[str, Any]:
    return {
        "x": int(round(x)),
        "y": int(round(y)),
        "nx": round(float(x) / max(width - 1, 1), 6),
        "ny": round(float(y) / max(height - 1, 1), 6),
    }


def _json_points(
    points: Iterable[Sequence[float]], width: int, height: int
) -> List[Dict[str, Any]]:
    return [_json_point(point[0], point[1], width, height) for point in points]


def _component_centerline(component: np.ndarray) -> np.ndarray:
    """沿畫面 y 軸取中位數，將一塊像素遮罩壓成中心線。"""
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    step = max(4, component.shape[0] // 120)
    points = []
    for y0 in range(int(ys.min()), int(ys.max()) + 1, step):
        selected = (ys >= y0) & (ys < y0 + step)
        if not selected.any():
            continue
        points.append((float(np.median(xs[selected])), float(np.mean(ys[selected]))))
    return np.asarray(points, dtype=np.float32)


def _simplify_contour(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.5, 0.006 * cv2.arcLength(contour, True))
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)


def _fit_polyline(points: np.ndarray, height: int) -> np.ndarray:
    """以 x=f(y) 擬合透視下的道路線，輸出由近到遠的連續座標。"""
    if len(points) < 2:
        return points
    ys = points[:, 1]
    xs = points[:, 0]
    unique_y = np.unique(np.round(ys).astype(int))
    degree = 2 if len(unique_y) >= 3 else 1
    try:
        coefficients = np.polyfit(ys, xs, degree)
    except (ValueError, np.linalg.LinAlgError):
        return points
    sample_count = max(2, min(80, int((ys.max() - ys.min()) / max(height / 120, 2))))
    sample_y = np.linspace(float(ys.max()), float(ys.min()), sample_count)
    sample_x = np.polyval(coefficients, sample_y)
    return np.column_stack((sample_x, sample_y)).astype(np.float32)


def _group_dashed_segments(
    segments: List[Dict[str, Any]], width: int, height: int
) -> List[np.ndarray]:
    """把虛線像素塊依延伸方向串成 lane；segments 本身仍完整保留。"""
    usable = [item for item in segments if len(item["raw_centerline"]) >= 2]
    usable.sort(key=lambda item: item["centroid"][1], reverse=True)
    tracks: List[List[Dict[str, Any]]] = []
    max_x_error = max(28.0, width * 0.055)
    max_y_gap = height * 0.32

    for segment in usable:
        sx, sy = segment["centroid"]
        best_track = None
        best_error = float("inf")
        for track in tracks:
            previous = track[-1]
            previous_y = previous["centroid"][1]
            if sy > previous_y + height * 0.03 or previous_y - sy > max_y_gap:
                continue
            centers = np.asarray([item["centroid"] for item in track], dtype=np.float32)
            if len(centers) >= 2 and np.ptp(centers[:, 1]) > 2:
                predicted_x = float(np.polyval(np.polyfit(centers[:, 1], centers[:, 0], 1), sy))
            else:
                predicted_x = float(previous["centroid"][0])
            error = abs(predicted_x - sx)
            if error < max_x_error and error < best_error:
                best_error = error
                best_track = track
        if best_track is None:
            tracks.append([segment])
        else:
            best_track.append(segment)

    polylines = []
    for track in tracks:
        all_points = np.concatenate([item["raw_centerline"] for item in track])
        polylines.append(_fit_polyline(all_points, height))
    return polylines


def extract_marking_coordinates(
    class_map: np.ndarray,
    class_ids: Sequence[int] = DEFAULT_CLASS_IDS,
    minimum_area: int = 45,
) -> List[Dict[str, Any]]:
    """將語意遮罩轉成可供其他程式使用的像素座標及 0~1 正規化座標。"""
    height, width = class_map.shape
    output = []

    for class_id in class_ids:
        binary = (class_map == class_id).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        components = []
        internal_segments = []

        for label_id in range(1, count):
            x, y, box_width, box_height, area = stats[label_id]
            scaled_minimum = max(minimum_area, int(width * height * 0.000015))
            if int(area) < scaled_minimum:
                continue
            component = labels == label_id
            centerline = _component_centerline(component)
            contour = _simplify_contour(component)
            centroid = (float(centroids[label_id][0]), float(centroids[label_id][1]))
            item = {
                "area_pixels": int(area),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(box_width),
                    "height": int(box_height),
                },
                "centroid": _json_point(centroid[0], centroid[1], width, height),
                "polygon": _json_points(contour, width, height),
                "centerline": _json_points(centerline, width, height),
            }
            components.append(item)
            internal_segments.append(
                {
                    "centroid": centroid,
                    "raw_centerline": centerline,
                }
            )

        if not components:
            continue

        marking = {
            "class_id": class_id,
            "class_name": ROAD_MARKING_CLASSES[class_id][0],
            "kind": "area" if class_id in AREA_CLASS_IDS else "line",
            "components": components,
        }
        if class_id in LINE_CLASS_IDS:
            if class_id in DASHED_CLASS_IDS:
                fitted = _group_dashed_segments(internal_segments, width, height)
            else:
                fitted = [
                    _fit_polyline(segment["raw_centerline"], height)
                    for segment in internal_segments
                    if len(segment["raw_centerline"]) >= 2
                ]
            marking["polylines"] = [
                {
                    "lane_id": index,
                    "points": _json_points(polyline, width, height),
                }
                for index, polyline in enumerate(fitted, start=1)
            ]
            if class_id in DASHED_CLASS_IDS:
                marking["coordinate_note"] = (
                    "components 是每段虛線；polylines 是將同車道虛線串接後的整條線"
                )
        output.append(marking)
    return output


def draw_road_markings(
    frame: np.ndarray,
    result: RoadMarkingResult,
    opacity: float = 0.38,
    class_ids: Sequence[int] = DEFAULT_CLASS_IDS,
) -> np.ndarray:
    """用已快取的分割結果疊圖，不需要保留 Stage-5 模型。"""
    output = frame.copy()
    color_mask = np.zeros_like(frame)
    selected = np.zeros(frame.shape[:2], dtype=bool)
    for class_id in class_ids:
        pixels = result.class_map == class_id
        color_mask[pixels] = ROAD_MARKING_CLASSES[class_id][1]
        selected |= pixels
    blended = cv2.addWeighted(frame, 1.0 - opacity, color_mask, opacity, 0)
    output[selected] = blended[selected]

    for marking in result.objects:
        color = ROAD_MARKING_CLASSES[marking["class_id"]][1]
        for polyline in marking.get("polylines", []):
            points = np.asarray(
                [(point["x"], point["y"]) for point in polyline["points"]],
                dtype=np.int32,
            )
            if len(points) >= 2:
                cv2.polylines(output, [points], False, color, 2, cv2.LINE_AA)
    return output


class RoadMarkingDetector:
    """以 Stage-5 SegFormer checkpoint 推論 RLMD 道路標線。"""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cuda",
        max_side: int = 1280,
        class_ids: Sequence[int] = DEFAULT_CLASS_IDS,
    ) -> None:
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise RuntimeError(
                "缺少 transformers；請在專案環境執行 pip install -r requirements.txt"
            ) from error

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到道路標線 checkpoint：{self.checkpoint_path}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("要求 CUDA，但 PyTorch 目前偵測不到 NVIDIA GPU")

        self.device = torch.device(device)
        self.max_side = max(512, int(max_side))
        self.class_ids = tuple(class_ids)
        config = SegformerConfig(num_labels=len(ROAD_MARKING_CLASSES))
        self.model = SegformerForSemanticSegmentation(config)
        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]

    def _input_size(self, width: int, height: int) -> Tuple[int, int]:
        scale = min(1.0, self.max_side / max(width, height))
        resized_width = max(32, int(round(width * scale / 32)) * 32)
        resized_height = max(32, int(round(height * scale / 32)) * 32)
        return resized_width, resized_height

    def _forward_crop(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=image).logits
        return functional.interpolate(
            logits, image.shape[-2:], mode="bilinear", align_corners=False
        )

    def _slide_inference(self, image: torch.Tensor) -> torch.Tensor:
        crop_height = crop_width = 512
        stride_height = stride_width = 384
        _, _, image_height, image_width = image.shape
        rows = max(image_height - crop_height + stride_height - 1, 0) // stride_height + 1
        columns = max(image_width - crop_width + stride_width - 1, 0) // stride_width + 1
        predictions = image.new_zeros((1, len(ROAD_MARKING_CLASSES), image_height, image_width))
        counts = image.new_zeros((1, 1, image_height, image_width))

        for row in range(rows):
            for column in range(columns):
                y2 = min(row * stride_height + crop_height, image_height)
                x2 = min(column * stride_width + crop_width, image_width)
                y1 = max(y2 - crop_height, 0)
                x1 = max(x2 - crop_width, 0)
                crop = image[:, :, y1:y2, x1:x2]
                crop_logits = self._forward_crop(crop)
                predictions[:, :, y1:y2, x1:x2] += crop_logits
                counts[:, :, y1:y2, x1:x2] += 1
        return predictions / counts.clamp_min(1)

    def predict(self, frame: np.ndarray) -> RoadMarkingResult:
        started_at = perf_counter()
        height, width = frame.shape[:2]
        input_width, input_height = self._input_size(width, height)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        image = image.to(self.device, dtype=torch.float32)[None, :] / 255.0
        image = (image - self.mean) / self.std

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits = self._slide_inference(image)
            class_map = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        class_map = cv2.resize(class_map, (width, height), interpolation=cv2.INTER_NEAREST)
        objects = extract_marking_coordinates(class_map, self.class_ids)
        return RoadMarkingResult(
            class_map=class_map,
            objects=objects,
            inference_ms=(perf_counter() - started_at) * 1000.0,
            model_input_size=(input_width, input_height),
        )

    def draw(
        self, frame: np.ndarray, result: RoadMarkingResult, opacity: float = 0.38
    ) -> np.ndarray:
        return draw_road_markings(frame, result, opacity, self.class_ids)
