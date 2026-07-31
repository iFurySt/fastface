from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fastface.evaluation.benchmark_widerface_detectors import parse_widerface_bbx
from fastface.pipeline.detectors import DetectedFace, build_detector
from fastface.pipeline.runtime import FastFaceOnnxPredictor, align_face


def box_iou_one(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-8)


def face_scale(face: DetectedFace) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(((x2 - x1) * (y2 - y1)) ** 0.5, 1.0)


def normalized_landmark_error(candidate: DetectedFace, baseline: DetectedFace) -> float | None:
    if not candidate.has_alignment_landmarks or not baseline.has_alignment_landmarks:
        return None
    candidate_points = np.asarray(candidate.landmarks, dtype=np.float32)
    baseline_points = np.asarray(baseline.landmarks, dtype=np.float32)
    distances = np.linalg.norm(candidate_points - baseline_points, axis=1)
    return float(distances.mean() / face_scale(baseline))


def aligned_crop_mae(image_bgr: np.ndarray, candidate: DetectedFace, baseline: DetectedFace, output_size: int) -> float | None:
    if not candidate.has_alignment_landmarks or not baseline.has_alignment_landmarks:
        return None
    candidate_crop, candidate_mode = align_face(image_bgr, candidate, output_size=output_size)
    baseline_crop, baseline_mode = align_face(image_bgr, baseline, output_size=output_size)
    if candidate_mode != "landmark_5pt" or baseline_mode != "landmark_5pt":
        return None
    candidate_float = candidate_crop.astype(np.float32)
    baseline_float = baseline_crop.astype(np.float32)
    return float(np.mean(np.abs(candidate_float - baseline_float)) / 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare owned detector alignment outputs against a baseline detector.")
    parser.add_argument("--widerface-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--fastface-model", type=Path, required=True)
    parser.add_argument("--fastface-input-size", type=int)
    parser.add_argument("--candidate-onnx", type=Path, required=True)
    parser.add_argument("--candidate-network", default="mobilenetv1_0.50")
    parser.add_argument("--candidate-input-size", type=int, default=1280)
    parser.add_argument("--candidate-resize-mode", choices=["square", "max-side"], default="max-side")
    parser.add_argument("--candidate-conf", type=float, default=0.55)
    parser.add_argument("--candidate-nms", type=float, default=0.3)
    parser.add_argument("--candidate-pre-nms-topk", type=int, default=1000)
    parser.add_argument("--candidate-post-nms-topk", type=int, default=750)
    parser.add_argument("--baseline-detector", choices=["retinaface", "scrfd"], default="retinaface")
    parser.add_argument("--baseline-model", default="retinaface_mnet_v2")
    parser.add_argument("--baseline-input-size", type=int, default=640)
    parser.add_argument("--baseline-conf", type=float, default=0.5)
    parser.add_argument("--baseline-nms", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_file = args.widerface_root / "wider_face_split" / f"wider_face_{args.split}_bbx_gt.txt"
    if not split_file.exists():
        split_file = args.widerface_root / f"wider_face_{args.split}_bbx_gt.txt"
    image_root = args.widerface_root / args.split / "images"
    items = parse_widerface_bbx(split_file, max_images=args.max_images)

    candidate = build_detector(
        backend="owned-retinaface-onnx",
        model_name=None,
        confidence_threshold=args.candidate_conf,
        nms_threshold=args.candidate_nms,
        input_size=args.candidate_input_size,
        providers=["CPUExecutionProvider"],
        owned_onnx_path=str(args.candidate_onnx),
        owned_network=args.candidate_network,
        resize_mode=args.candidate_resize_mode,
        pre_nms_topk=args.candidate_pre_nms_topk,
        post_nms_topk=args.candidate_post_nms_topk,
    )
    baseline = build_detector(
        backend=args.baseline_detector,
        model_name=args.baseline_model,
        confidence_threshold=args.baseline_conf,
        nms_threshold=args.baseline_nms,
        input_size=args.baseline_input_size,
        providers=["CPUExecutionProvider"],
    )
    predictor = FastFaceOnnxPredictor(model_path=args.fastface_model, input_size=args.fastface_input_size)

    images = 0
    matched_faces = 0
    candidate_faces_total = 0
    baseline_faces_total = 0
    landmark_errors: list[float] = []
    crop_errors: list[float] = []
    age_diffs: list[float] = []
    gender_agreements = 0
    ious: list[float] = []
    missing_images: list[str] = []

    for item in items:
        image_path = image_root / item.image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            missing_images.append(str(image_path))
            continue
        images += 1
        candidate_faces = candidate.detect(image, max_faces=0, selection_metric="max")
        baseline_faces = baseline.detect(image, max_faces=0, selection_metric="max")
        candidate_faces_total += len(candidate_faces)
        baseline_faces_total += len(baseline_faces)
        used_baseline: set[int] = set()
        for candidate_face in candidate_faces:
            best_index = -1
            best_iou = 0.0
            for index, baseline_face in enumerate(baseline_faces):
                if index in used_baseline:
                    continue
                iou = box_iou_one(candidate_face.bbox, baseline_face.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index < 0 or best_iou < args.iou_threshold:
                continue
            baseline_face = baseline_faces[best_index]
            used_baseline.add(best_index)
            matched_faces += 1
            ious.append(best_iou)
            landmark_error = normalized_landmark_error(candidate_face, baseline_face)
            if landmark_error is not None:
                landmark_errors.append(landmark_error)
            crop_error = aligned_crop_mae(image, candidate_face, baseline_face, output_size=predictor.input_size)
            if crop_error is not None:
                crop_errors.append(crop_error)
            candidate_crop, _ = align_face(image, candidate_face, predictor.input_size)
            baseline_crop, _ = align_face(image, baseline_face, predictor.input_size)
            candidate_prediction = predictor.predict(candidate_crop)
            baseline_prediction = predictor.predict(baseline_crop)
            age_diffs.append(abs(candidate_prediction.age - baseline_prediction.age))
            if candidate_prediction.gender == baseline_prediction.gender:
                gender_agreements += 1

    result = {
        "candidate": candidate.name,
        "baseline": baseline.name,
        "images": images,
        "missing_images": missing_images[:20],
        "candidate_faces": candidate_faces_total,
        "baseline_faces": baseline_faces_total,
        "matched_faces": matched_faces,
        "match_rate_vs_baseline": matched_faces / max(baseline_faces_total, 1),
        "mean_bbox_iou": float(np.mean(ious)) if ious else None,
        "mean_normalized_landmark_error": float(np.mean(landmark_errors)) if landmark_errors else None,
        "p95_normalized_landmark_error": float(np.percentile(landmark_errors, 95)) if landmark_errors else None,
        "mean_aligned_crop_mae": float(np.mean(crop_errors)) if crop_errors else None,
        "p95_aligned_crop_mae": float(np.percentile(crop_errors, 95)) if crop_errors else None,
        "mean_age_abs_diff": float(np.mean(age_diffs)) if age_diffs else None,
        "p95_age_abs_diff": float(np.percentile(age_diffs, 95)) if age_diffs else None,
        "gender_agreement": gender_agreements / max(matched_faces, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
