from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Protocol

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class WiderFaceItem:
    image: str
    boxes: np.ndarray


class Detector(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        ...


def parse_widerface_bbx(path: Path, max_images: int | None = None) -> list[WiderFaceItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[WiderFaceItem] = []
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
            boxes.append([x, y, x + width, y + height])
        items.append(WiderFaceItem(image=image_name, boxes=np.asarray(boxes, dtype=np.float32)))
        if max_images is not None and len(items) >= max_images:
            break
    return items


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
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


def greedy_match(pred_boxes: np.ndarray, target_boxes: np.ndarray, iou_threshold: float) -> tuple[int, int, int]:
    if pred_boxes.size == 0:
        return 0, 0, int(target_boxes.shape[0])
    if target_boxes.size == 0:
        return 0, int(pred_boxes.shape[0]), 0
    ious = box_iou(pred_boxes, target_boxes)
    pairs = [
        (float(ious[pred_idx, target_idx]), pred_idx, target_idx)
        for pred_idx in range(ious.shape[0])
        for target_idx in range(ious.shape[1])
        if ious[pred_idx, target_idx] >= iou_threshold
    ]
    pairs.sort(reverse=True)
    used_preds: set[int] = set()
    used_targets: set[int] = set()
    true_positive = 0
    for _, pred_idx, target_idx in pairs:
        if pred_idx in used_preds or target_idx in used_targets:
            continue
        used_preds.add(pred_idx)
        used_targets.add(target_idx)
        true_positive += 1
    false_positive = int(pred_boxes.shape[0]) - true_positive
    false_negative = int(target_boxes.shape[0]) - true_positive
    return true_positive, false_positive, false_negative


class UnifaceDetector:
    def __init__(self, backend: str, model_name: str | None, confidence_threshold: float, nms_threshold: float, input_size: int) -> None:
        from fastface.pipeline.detectors import build_detector

        self.detector = build_detector(
            backend=backend,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            providers=["CPUExecutionProvider"],
        )
        self.name = self.detector.name

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        faces = self.detector.detect(image_bgr, max_faces=0, selection_metric="max")
        return np.asarray([face.bbox for face in faces], dtype=np.float32)


class OwnedRetinaFaceDetector:
    def __init__(
        self,
        repo_path: Path,
        weights_path: Path,
        network: str,
        confidence_threshold: float,
        nms_threshold: float,
        pre_nms_topk: int,
        post_nms_topk: int,
        device_name: str,
    ) -> None:
        self.repo_path = repo_path
        self.weights_path = weights_path
        self.network = network
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.name = f"owned-retinaface:{network}:{weights_path.name}"

        sys.path.insert(0, str(repo_path))
        from config import get_config
        from layers import PriorBox
        from models import RetinaFace
        from utils.box_utils import decode, decode_landmarks, nms

        self.get_config = get_config
        self.PriorBox = PriorBox
        self.decode = decode
        self.decode_landmarks = decode_landmarks
        self.nms = nms
        self.cfg = get_config(network)
        if self.cfg is None:
            raise ValueError(f"unsupported owned RetinaFace network: {network}")
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)
        self.model = RetinaFace(cfg=self.cfg)
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        image = np.float32(image_bgr)
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        height, width = image_bgr.shape[:2]
        with torch.no_grad():
            loc, conf, landmarks = self.model(tensor)
        loc = loc.squeeze(0)
        conf = conf.squeeze(0)
        landmarks = landmarks.squeeze(0)
        priors = self.PriorBox(self.cfg, image_size=(height, width)).generate_anchors().to(self.device)
        boxes = self.decode(loc, priors, self.cfg["variance"])
        landmarks = self.decode_landmarks(landmarks, priors, self.cfg["variance"])
        bbox_scale = torch.tensor([width, height] * 2, device=self.device)
        boxes = (boxes * bbox_scale).cpu().numpy()
        scores = conf[:, 1].detach().cpu().numpy()
        inds = scores > self.confidence_threshold
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1][: self.pre_nms_topk]
        boxes = boxes[order]
        scores = scores[order]
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.nms(detections, self.nms_threshold)
        detections = detections[keep][: self.post_nms_topk]
        return detections[:, :4].astype(np.float32)


def evaluate_detector(
    detector: Detector,
    items: list[WiderFaceItem],
    image_root: Path,
    iou_threshold: float,
) -> dict[str, object]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    image_count = 0
    elapsed = 0.0
    missing_images: list[str] = []
    for item in items:
        image_path = image_root / item.image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            missing_images.append(str(image_path))
            continue
        start = time.perf_counter()
        pred_boxes = detector.detect(image)
        elapsed += time.perf_counter() - start
        tp, fp, fn = greedy_match(pred_boxes, item.boxes, iou_threshold=iou_threshold)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        image_count += 1
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "detector": detector.name,
        "images": image_count,
        "missing_images": missing_images[:20],
        "iou_threshold": iou_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_seconds": elapsed,
        "seconds_per_image": elapsed / max(image_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark face detectors on WIDER FACE bbox annotations.")
    parser.add_argument("--widerface-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--detector", choices=["retinaface", "scrfd", "owned-retinaface"], default="retinaface")
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-input-size", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-nms", type=float, default=0.4)
    parser.add_argument("--pre-nms-topk", type=int, default=5000)
    parser.add_argument("--post-nms-topk", type=int, default=750)
    parser.add_argument("--owned-retinaface-repo", type=Path)
    parser.add_argument("--owned-retinaface-weights", type=Path)
    parser.add_argument("--owned-retinaface-network", default="mobilenetv1_0.50")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_file = args.widerface_root / "wider_face_split" / f"wider_face_{args.split}_bbx_gt.txt"
    if not split_file.exists():
        split_file = args.widerface_root / f"wider_face_{args.split}_bbx_gt.txt"
    image_root = args.widerface_root / args.split / "images"
    items = parse_widerface_bbx(split_file, max_images=args.max_images)
    if args.detector == "owned-retinaface":
        if args.owned_retinaface_repo is None or args.owned_retinaface_weights is None:
            raise ValueError("--owned-retinaface-repo and --owned-retinaface-weights are required")
        detector: Detector = OwnedRetinaFaceDetector(
            repo_path=args.owned_retinaface_repo,
            weights_path=args.owned_retinaface_weights,
            network=args.owned_retinaface_network,
            confidence_threshold=args.detector_conf,
            nms_threshold=args.detector_nms,
            pre_nms_topk=args.pre_nms_topk,
            post_nms_topk=args.post_nms_topk,
            device_name=args.device,
        )
    else:
        detector = UnifaceDetector(
            backend=args.detector,
            model_name=args.detector_model,
            confidence_threshold=args.detector_conf,
            nms_threshold=args.detector_nms,
            input_size=args.detector_input_size,
        )
    result = evaluate_detector(detector, items=items, image_root=image_root, iou_threshold=args.iou_threshold)
    result["split_file"] = str(split_file)
    result["image_root"] = str(image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
