from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]
    score: float
    landmarks: tuple[tuple[float, float], ...]

    @property
    def has_alignment_landmarks(self) -> bool:
        return len(self.landmarks) == 5


class FaceDetector(Protocol):
    name: str

    def detect(
        self,
        image_bgr: np.ndarray,
        max_faces: int,
        selection_metric: str,
    ) -> list[DetectedFace]:
        ...


class UnifaceDetector:
    def __init__(
        self,
        backend: str,
        model_name: str | None,
        confidence_threshold: float,
        nms_threshold: float,
        input_size: int,
        providers: list[str] | None = None,
    ) -> None:
        try:
            from uniface.constants import RetinaFaceWeights, SCRFDWeights
            from uniface.detection import RetinaFace, SCRFD
        except ImportError as exc:
            raise RuntimeError(
                "UniFace is required for this detector backend. Install it with "
                "`python -m pip install 'uniface[cpu]'` or use a future owned FastFace detector backend."
            ) from exc

        self.backend = backend
        self.name = f"uniface:{backend}"

        if backend == "retinaface":
            selected_model = self._enum_by_value(RetinaFaceWeights, model_name) if model_name else RetinaFaceWeights.MNET_V2
            self._detector = RetinaFace(
                model_name=selected_model,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                input_size=(input_size, input_size),
                providers=providers,
            )
            self.name = f"uniface:retinaface:{selected_model.value}"
            return

        if backend == "scrfd":
            selected_model = self._enum_by_value(SCRFDWeights, model_name) if model_name else SCRFDWeights.SCRFD_10G_KPS
            self._detector = SCRFD(
                model_name=selected_model,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                input_size=(input_size, input_size),
                providers=providers,
            )
            self.name = f"uniface:scrfd:{selected_model.value}"
            return

        raise ValueError(f"unsupported UniFace detector backend: {backend}")

    @staticmethod
    def _enum_by_value(enum_type: type, value: str) -> object:
        for item in enum_type:
            if item.value == value or item.name.lower() == value.lower():
                return item
        allowed = ", ".join(f"{item.name}/{item.value}" for item in enum_type)
        raise ValueError(f"unsupported detector model {value!r}; allowed: {allowed}")

    def detect(
        self,
        image_bgr: np.ndarray,
        max_faces: int,
        selection_metric: str,
    ) -> list[DetectedFace]:
        faces = self._detector.detect(
            image_bgr,
            max_num=max_faces,
            metric=selection_metric,
        )
        return [
            DetectedFace(
                bbox=tuple(float(v) for v in face.bbox[:4]),
                score=float(face.confidence),
                landmarks=tuple((float(x), float(y)) for x, y in face.landmarks),
            )
            for face in faces
        ]


RETINAFACE_CONFIGS: dict[str, dict[str, object]] = {
    "mobilenetv1_0.50": {
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "steps": [8, 16, 32],
        "variance": [0.1, 0.2],
        "clip": False,
    },
}


class OwnedRetinaFaceOnnxDetector:
    def __init__(
        self,
        model_path: str,
        network: str,
        confidence_threshold: float,
        nms_threshold: float,
        input_size: int,
        resize_mode: str,
        pre_nms_topk: int,
        post_nms_topk: int,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is required for the owned detector backend") from exc

        if network not in RETINAFACE_CONFIGS:
            allowed = ", ".join(sorted(RETINAFACE_CONFIGS))
            raise ValueError(f"unsupported owned RetinaFace network {network!r}; allowed: {allowed}")
        if resize_mode not in {"square", "max-side"}:
            raise ValueError(f"unsupported detector resize mode: {resize_mode}")

        self.model_path = model_path
        self.network = network
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.resize_mode = resize_mode
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.cfg = RETINAFACE_CONFIGS[network]
        self._prior_cache: dict[tuple[int, int], np.ndarray] = {}
        self.name = f"owned-retinaface-onnx:{network}"

        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(
        self,
        image_bgr: np.ndarray,
        max_faces: int,
        selection_metric: str,
    ) -> list[DetectedFace]:
        original_height, original_width = image_bgr.shape[:2]
        image, resize_factor = resize_for_detector(image_bgr, input_size=self.input_size, resize_mode=self.resize_mode)
        height, width = image.shape[:2]
        batch = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(0, 0),
            mean=(104, 117, 123),
            swapRB=False,
            crop=False,
        )
        outputs = self.session.run(self.output_names, {self.input_name: batch})
        loc = np.asarray(outputs[0], dtype=np.float32).squeeze(0)
        conf = np.asarray(outputs[1], dtype=np.float32).squeeze(0)
        landmarks = np.asarray(outputs[2], dtype=np.float32).squeeze(0)
        priors = self._get_priors(height=height, width=width)

        boxes = decode_retinaface_boxes(loc, priors, self.cfg["variance"])
        decoded_landmarks = decode_retinaface_landmarks(landmarks, priors, self.cfg["variance"])
        bbox_scale = np.asarray([width, height] * 2, dtype=np.float32)
        landmark_scale = np.asarray([width, height] * 5, dtype=np.float32)
        boxes = boxes * bbox_scale / resize_factor
        decoded_landmarks = decoded_landmarks * landmark_scale / resize_factor
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_height)
        decoded_landmarks[:, 0::2] = np.clip(decoded_landmarks[:, 0::2], 0, original_width)
        decoded_landmarks[:, 1::2] = np.clip(decoded_landmarks[:, 1::2], 0, original_height)

        scores = conf[:, 1]
        keep = scores > self.confidence_threshold
        boxes = boxes[keep]
        decoded_landmarks = decoded_landmarks[keep]
        scores = scores[keep]
        order = scores.argsort()[::-1][: self.pre_nms_topk]
        boxes = boxes[order]
        decoded_landmarks = decoded_landmarks[order]
        scores = scores[order]

        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep_indices = nms(detections, self.nms_threshold)[: self.post_nms_topk]
        detections = detections[keep_indices]
        decoded_landmarks = decoded_landmarks[keep_indices]
        if max_faces > 0:
            detections = detections[:max_faces]
            decoded_landmarks = decoded_landmarks[:max_faces]

        return [
            DetectedFace(
                bbox=tuple(float(value) for value in detection[:4]),
                score=float(detection[4]),
                landmarks=tuple(
                    (float(points[index]), float(points[index + 1]))
                    for index in range(0, 10, 2)
                ),
            )
            for detection, points in zip(detections, decoded_landmarks)
        ]

    def _get_priors(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        priors = self._prior_cache.get(key)
        if priors is None:
            priors = generate_retinaface_priors(self.cfg, height=height, width=width)
            self._prior_cache[key] = priors
        return priors


def resize_for_detector(image_bgr: np.ndarray, input_size: int, resize_mode: str) -> tuple[np.ndarray, float]:
    height, width = image_bgr.shape[:2]
    if resize_mode == "square":
        resize_factor = input_size / max(height, width)
        resized = cv2.resize(
            image_bgr,
            (int(round(width * resize_factor)), int(round(height * resize_factor))),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros((input_size, input_size, 3), dtype=image_bgr.dtype)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        return canvas, resize_factor
    if resize_mode == "max-side":
        resize_factor = min(1.0, input_size / max(height, width))
        if resize_factor == 1.0:
            return image_bgr, 1.0
        resized = cv2.resize(
            image_bgr,
            (int(round(width * resize_factor)), int(round(height * resize_factor))),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, resize_factor
    raise ValueError(f"unsupported detector resize mode: {resize_mode}")


def generate_retinaface_priors(cfg: dict[str, object], height: int, width: int) -> np.ndarray:
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


def decode_retinaface_boxes(loc: np.ndarray, priors: np.ndarray, variances: object) -> np.ndarray:
    variance_values = list(variances)
    centers = priors[:, :2] + loc[:, :2] * variance_values[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(loc[:, 2:] * variance_values[1])
    boxes = np.empty_like(loc, dtype=np.float32)
    boxes[:, :2] = centers - sizes / 2
    boxes[:, 2:] = centers + sizes / 2
    return boxes


def decode_retinaface_landmarks(landmarks: np.ndarray, priors: np.ndarray, variances: object) -> np.ndarray:
    variance_values = list(variances)
    points = landmarks.reshape(-1, 5, 2)
    decoded = priors[:, :2, np.newaxis].transpose(0, 2, 1) + points * variance_values[0] * priors[:, 2:, np.newaxis].transpose(0, 2, 1)
    return decoded.reshape(-1, 10).astype(np.float32)


def nms(detections: np.ndarray, threshold: float) -> list[int]:
    if detections.size == 0:
        return []
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        overlap = intersection / (areas[index] + areas[order[1:]] - intersection)
        indices = np.where(overlap <= threshold)[0]
        order = order[indices + 1]
    return keep


def build_detector(
    backend: str,
    model_name: str | None,
    confidence_threshold: float,
    nms_threshold: float,
    input_size: int,
    providers: list[str] | None,
    owned_onnx_path: str | None = None,
    owned_network: str = "mobilenetv1_0.50",
    resize_mode: str = "square",
    pre_nms_topk: int = 1000,
    post_nms_topk: int = 750,
) -> FaceDetector:
    if backend in {"retinaface", "scrfd"}:
        return UnifaceDetector(
            backend=backend,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            providers=providers,
        )
    if backend == "owned-retinaface-onnx":
        if owned_onnx_path is None:
            raise ValueError("--owned-detector-onnx is required for owned-retinaface-onnx")
        return OwnedRetinaFaceOnnxDetector(
            model_path=owned_onnx_path,
            network=owned_network,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            resize_mode=resize_mode,
            pre_nms_topk=pre_nms_topk,
            post_nms_topk=post_nms_topk,
            providers=providers,
        )
    raise ValueError(f"unsupported detector backend: {backend}")
