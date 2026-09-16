import argparse
import json
from pathlib import Path

import cv2
import torch

from road_marking import RoadMarkingDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "road_marking" / "stage5_roadmark_best.pth"
)


def main():
    parser = argparse.ArgumentParser(
        description="使用 Stage-5 測試單張圖片並輸出道路標線座標"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "road_marking_results")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-side", type=int, default=1280)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"找不到圖片：{args.image}")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f"OpenCV 無法讀取圖片：{args.image}")

    detector = RoadMarkingDetector(
        checkpoint_path=args.checkpoint,
        device=device,
        max_side=args.max_side,
    )
    result = detector.predict(frame)
    overlay = detector.draw(frame, result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    image_output = args.output_dir / f"{stem}_stage5.jpg"
    json_output = args.output_dir / f"{stem}_stage5.json"
    cv2.imwrite(str(image_output), overlay)
    json_output.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"裝置：{device}")
    print(f"推論時間：{result.inference_ms:.0f} ms")
    print(f"辨識類別數：{len(result.objects)}")
    print(f"疊圖：{image_output}")
    print(f"座標：{json_output}")


if __name__ == "__main__":
    main()
