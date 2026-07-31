from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_widerface_bbx(path: Path, max_images: int | None = None) -> list[tuple[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[tuple[str, np.ndarray]] = []
    idx = 0
    while idx < len(lines):
        image_name = lines[idx].strip()
        idx += 1
        if not image_name:
            continue
        face_count = int(lines[idx].strip())
        idx += 1
        annotation_rows = face_count if face_count > 0 else 1
        boxes: list[list[float]] = []
        for _ in range(annotation_rows):
            parts = [float(value) for value in lines[idx].strip().split()]
            idx += 1
            if face_count == 0:
                continue
            x, y, width, height = parts[:4]
            if width <= 1 or height <= 1:
                continue
            boxes.append([x, y, width, height])
        items.append((image_name, np.asarray(boxes, dtype=np.float32)))
        if max_images is not None and len(items) >= max_images:
            break
    return items


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)


def fallback_points(x: float, y: float, width: float, height: float) -> list[tuple[float, float]]:
    return [
        (x + 0.35 * width, y + 0.40 * height),
        (x + 0.65 * width, y + 0.40 * height),
        (x + 0.50 * width, y + 0.55 * height),
        (x + 0.38 * width, y + 0.75 * height),
        (x + 0.62 * width, y + 0.75 * height),
    ]


def build_detector(model_name: str, confidence_threshold: float, nms_threshold: float, input_size: int) -> Any:
    from uniface.constants import RetinaFaceWeights
    from uniface.detection import RetinaFace

    selected = None
    for item in RetinaFaceWeights:
        if item.value == model_name or item.name.lower() == model_name.lower():
            selected = item
            break
    if selected is None:
        raise ValueError(f"unknown RetinaFace model: {model_name}")
    return RetinaFace(
        model_name=selected,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        input_size=(input_size, input_size),
        providers=["CPUExecutionProvider"],
    )


def convert(
    source_path: Path,
    image_root: Path,
    output_path: Path,
    model_name: str,
    confidence_threshold: float,
    nms_threshold: float,
    input_size: int,
    match_iou: float,
    max_images: int | None,
    drop_unmatched: bool,
) -> dict[str, int]:
    detector = build_detector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        input_size=input_size,
    )
    items = parse_widerface_bbx(source_path, max_images=max_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "images": 0,
        "faces": 0,
        "teacher_matched_faces": 0,
        "fallback_faces": 0,
        "missing_images": 0,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for image_name, boxes_xywh in items:
            if boxes_xywh.size == 0:
                continue
            image_path = image_root / image_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                stats["missing_images"] += 1
                continue
            faces = detector.detect(image)
            teacher_boxes = np.asarray([face.bbox[:4] for face in faces], dtype=np.float32)
            teacher_landmarks = [np.asarray(face.landmarks, dtype=np.float32) for face in faces]
            overlaps = iou_matrix(xywh_to_xyxy(boxes_xywh), teacher_boxes)
            handle.write(f"# {image_name}\n")
            stats["images"] += 1
            for box_index, (x, y, width, height) in enumerate(boxes_xywh):
                point_source = "fallback"
                points = fallback_points(float(x), float(y), float(width), float(height))
                if overlaps.shape[1] > 0:
                    teacher_index = int(np.argmax(overlaps[box_index]))
                    if overlaps[box_index, teacher_index] >= match_iou:
                        points = [(float(px), float(py)) for px, py in teacher_landmarks[teacher_index]]
                        point_source = "teacher"
                if point_source != "teacher" and drop_unmatched:
                    stats["fallback_faces"] += 1
                    continue
                row = [float(x), float(y), float(width), float(height)]
                for point_x, point_y in points:
                    row.extend([point_x, point_y, 0.0])
                handle.write(" ".join(f"{value:.3f}" for value in row) + "\n")
                stats["faces"] += 1
                if point_source == "teacher":
                    stats["teacher_matched_faces"] += 1
                else:
                    stats["fallback_faces"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RetinaFace labels using UniFace RetinaFace as a landmark teacher.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-model", default="retinaface_mnet_v2")
    parser.add_argument("--teacher-conf", type=float, default=0.5)
    parser.add_argument("--teacher-nms", type=float, default=0.4)
    parser.add_argument("--teacher-input-size", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--drop-unmatched", action="store_true", help="Drop GT boxes that do not match a teacher detection.")
    args = parser.parse_args()

    stats = convert(
        source_path=args.source,
        image_root=args.image_root,
        output_path=args.output,
        model_name=args.teacher_model,
        confidence_threshold=args.teacher_conf,
        nms_threshold=args.teacher_nms,
        input_size=args.teacher_input_size,
        match_iou=args.match_iou,
        max_images=args.max_images,
        drop_unmatched=args.drop_unmatched,
    )
    print(stats)


if __name__ == "__main__":
    main()
