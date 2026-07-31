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
        """Return detections as Nx5 [x1, y1, x2, y2, score]."""
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
    def __init__(
        self,
        backend: str,
        model_name: str | None,
        confidence_threshold: float,
        nms_threshold: float,
        input_size: int,
        resize_mode: str,
    ) -> None:
        if resize_mode != "square":
            raise ValueError("UniFace detector benchmarks only support square input resizing")

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
        if not faces:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray([[*face.bbox, face.score] for face in faces], dtype=np.float32)


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
        return detections[:, :5].astype(np.float32)


class OwnedRetinaFaceOnnxDetector:
    def __init__(
        self,
        repo_path: Path,
        model_path: Path,
        network: str,
        confidence_threshold: float,
        nms_threshold: float,
        pre_nms_topk: int,
        post_nms_topk: int,
        input_size: int | None,
        resize_mode: str,
    ) -> None:
        self.repo_path = repo_path
        self.model_path = model_path
        self.network = network
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.input_size = input_size
        self.resize_mode = resize_mode
        self._prior_cache: dict[tuple[int, int], np.ndarray] = {}
        self._profile_seconds: dict[str, float] = {
            "resize": 0.0,
            "preprocess": 0.0,
            "onnxruntime": 0.0,
            "prior": 0.0,
            "decode": 0.0,
            "filter_sort": 0.0,
            "nms": 0.0,
        }
        self._profile_images = 0
        self.name = f"owned-retinaface-onnx:{network}:{model_path.name}"

        sys.path.insert(0, str(repo_path))
        from config import get_config
        from utils.box_utils import nms

        self.nms = nms
        self.cfg = get_config(network)
        if self.cfg is None:
            raise ValueError(f"unsupported owned RetinaFace network: {network}")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        original_height, original_width = image_bgr.shape[:2]
        stage_start = time.perf_counter()
        if self.input_size is not None and self.input_size > 0:
            if self.resize_mode == "square":
                image, resize_factor = resize_image_to_square(image_bgr, self.input_size)
            elif self.resize_mode == "max-side":
                image, resize_factor = resize_image_max_side(image_bgr, self.input_size)
            else:
                raise ValueError(f"unsupported resize mode: {self.resize_mode}")
        else:
            image = image_bgr
            resize_factor = 1.0
        self._profile_seconds["resize"] += time.perf_counter() - stage_start
        height, width = image.shape[:2]
        stage_start = time.perf_counter()
        image = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(0, 0),
            mean=(104, 117, 123),
            swapRB=False,
            crop=False,
        )
        self._profile_seconds["preprocess"] += time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: image})
        self._profile_seconds["onnxruntime"] += time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        loc = np.asarray(outputs[0], dtype=np.float32).squeeze(0)
        conf = np.asarray(outputs[1]).squeeze(0)
        priors = self._get_priors(height=height, width=width)
        self._profile_seconds["prior"] += time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        boxes = decode_retinaface_boxes(loc, priors, self.cfg["variance"])
        bbox_scale = np.asarray([width, height] * 2, dtype=np.float32)
        boxes = boxes * bbox_scale / resize_factor
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_height)
        scores = conf[:, 1]
        self._profile_seconds["decode"] += time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        inds = scores > self.confidence_threshold
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1][: self.pre_nms_topk]
        boxes = boxes[order]
        scores = scores[order]
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        self._profile_seconds["filter_sort"] += time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        keep = self.nms(detections, self.nms_threshold)
        detections = detections[keep][: self.post_nms_topk]
        self._profile_seconds["nms"] += time.perf_counter() - stage_start
        self._profile_images += 1
        return detections[:, :5].astype(np.float32)

    def _get_priors(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        priors = self._prior_cache.get(key)
        if priors is None:
            priors = generate_retinaface_priors(self.cfg, height=height, width=width)
            self._prior_cache[key] = priors
        return priors

    def profile_summary(self) -> dict[str, object]:
        images = max(self._profile_images, 1)
        return {
            "images": self._profile_images,
            "total_seconds": dict(self._profile_seconds),
            "seconds_per_image": {
                key: value / images for key, value in self._profile_seconds.items()
            },
        }


def resize_image_to_square(image_bgr: np.ndarray, input_size: int) -> tuple[np.ndarray, float]:
    height, width = image_bgr.shape[:2]
    resize_factor = input_size / max(height, width)
    resized = cv2.resize(
        image_bgr,
        (int(round(width * resize_factor)), int(round(height * resize_factor))),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros((input_size, input_size, 3), dtype=image_bgr.dtype)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, resize_factor


def resize_image_max_side(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image_bgr.shape[:2]
    resize_factor = min(1.0, max_side / max(height, width))
    if resize_factor == 1.0:
        return image_bgr, 1.0
    resized = cv2.resize(
        image_bgr,
        (int(round(width * resize_factor)), int(round(height * resize_factor))),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, resize_factor


def generate_retinaface_priors(cfg: dict, height: int, width: int) -> np.ndarray:
    anchors: list[np.ndarray] = []
    for min_sizes, step in zip(cfg["min_sizes"], cfg["steps"]):
        feature_height = int(np.ceil(height / step))
        feature_width = int(np.ceil(width / step))
        rows = (np.arange(feature_height, dtype=np.float32) + 0.5) * step / height
        cols = (np.arange(feature_width, dtype=np.float32) + 0.5) * step / width
        grid_x, grid_y = np.meshgrid(cols, rows)
        centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        sizes = np.asarray([[min_size / width, min_size / height] for min_size in min_sizes], dtype=np.float32)
        anchors.append(
            np.concatenate(
                [
                    np.repeat(centers, len(min_sizes), axis=0),
                    np.tile(sizes, (centers.shape[0], 1)),
                ],
                axis=1,
            )
        )
    priors = np.concatenate(anchors, axis=0).astype(np.float32)
    if cfg.get("clip", False):
        np.clip(priors, 0.0, 1.0, out=priors)
    return priors


def decode_retinaface_boxes(loc: np.ndarray, priors: np.ndarray, variances: list[float]) -> np.ndarray:
    centers = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes = np.empty_like(loc, dtype=np.float32)
    boxes[:, :2] = centers - sizes / 2
    boxes[:, 2:] = centers + sizes / 2
    return boxes


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
        tp, fp, fn = greedy_match(pred_boxes[:, :4] if pred_boxes.size else pred_boxes.reshape(0, 4), item.boxes, iou_threshold=iou_threshold)
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
    parser.add_argument(
        "--detector",
        choices=["retinaface", "scrfd", "owned-retinaface", "owned-retinaface-onnx"],
        default="retinaface",
    )
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-input-size", type=int, default=640)
    parser.add_argument("--resize-mode", choices=["square", "max-side"], default="square")
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-nms", type=float, default=0.4)
    parser.add_argument("--pre-nms-topk", type=int, default=5000)
    parser.add_argument("--post-nms-topk", type=int, default=750)
    parser.add_argument("--owned-retinaface-repo", type=Path)
    parser.add_argument("--owned-retinaface-weights", type=Path)
    parser.add_argument("--owned-retinaface-onnx", type=Path)
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
    elif args.detector == "owned-retinaface-onnx":
        if args.owned_retinaface_repo is None or args.owned_retinaface_onnx is None:
            raise ValueError("--owned-retinaface-repo and --owned-retinaface-onnx are required")
        detector = OwnedRetinaFaceOnnxDetector(
            repo_path=args.owned_retinaface_repo,
            model_path=args.owned_retinaface_onnx,
            network=args.owned_retinaface_network,
            confidence_threshold=args.detector_conf,
            nms_threshold=args.detector_nms,
            pre_nms_topk=args.pre_nms_topk,
            post_nms_topk=args.post_nms_topk,
            input_size=args.detector_input_size,
            resize_mode=args.resize_mode,
        )
    else:
        detector = UnifaceDetector(
            backend=args.detector,
            model_name=args.detector_model,
            confidence_threshold=args.detector_conf,
            nms_threshold=args.detector_nms,
            input_size=args.detector_input_size,
            resize_mode=args.resize_mode,
        )
    result = evaluate_detector(detector, items=items, image_root=image_root, iou_threshold=args.iou_threshold)
    if hasattr(detector, "profile_summary"):
        result["detector_profile"] = detector.profile_summary()
    result["split_file"] = str(split_file)
    result["image_root"] = str(image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
