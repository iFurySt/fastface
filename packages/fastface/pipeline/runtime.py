from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from fastface.paths import expand_path
from fastface.pipeline.detectors import DetectedFace, FaceDetector

GENDER_NAMES = {
    0: "female",
    1: "male",
}


@dataclass(frozen=True)
class FastFacePrediction:
    gender: int
    gender_name: str
    female_prob: float
    male_prob: float
    gender_confidence: float
    age: float


def softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True).clip(min=1e-8)


def load_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def infer_input_size(model_path: Path, metadata: dict[str, Any]) -> int:
    if "input_size" in metadata:
        return int(metadata["input_size"])

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    for value in reversed(shape):
        if isinstance(value, int) and value > 0:
            return int(value)
    raise ValueError("input size is not in metadata and could not be inferred from ONNX input shape")


class FastFaceOnnxPredictor:
    def __init__(
        self,
        model_path: Path,
        input_size: int | None = None,
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 0,
    ) -> None:
        self.model_path = expand_path(model_path)
        self.metadata = load_metadata(self.model_path)
        self.input_size = int(input_size) if input_size is not None else infer_input_size(self.model_path, self.metadata)
        normalization = self.metadata.get("normalization", {})
        self.mean = np.asarray(normalization.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(normalization.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads > 0:
            options.intra_op_num_threads = intra_op_num_threads
        if inter_op_num_threads > 0:
            options.inter_op_num_threads = inter_op_num_threads
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        return np.transpose(normalized, (2, 0, 1))[np.newaxis].astype(np.float32)

    def predict(self, face_bgr: np.ndarray) -> FastFacePrediction:
        batch = self.preprocess(face_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: batch})
        by_name = {name: output for name, output in zip(self.output_names, outputs)}

        if "gender_logits" in by_name:
            gender_logits = np.asarray(by_name["gender_logits"], dtype=np.float32)
        else:
            gender_logits = np.asarray(outputs[0], dtype=np.float32)

        if "age" in by_name:
            age_value = float(np.asarray(by_name["age"]).reshape(-1)[0])
        else:
            age_logits = np.asarray(by_name.get("age_logits", outputs[1]), dtype=np.float32)
            age_probs = softmax(age_logits, axis=1)
            age_value = float((age_probs * np.arange(age_probs.shape[1], dtype=np.float32)).sum(axis=1)[0])

        gender_probs = softmax(gender_logits, axis=1)[0]
        gender = int(np.argmax(gender_probs))
        return FastFacePrediction(
            gender=gender,
            gender_name=GENDER_NAMES.get(gender, str(gender)),
            female_prob=float(gender_probs[0]),
            male_prob=float(gender_probs[1]),
            gender_confidence=float(np.max(gender_probs)),
            age=age_value,
        )


def align_face(image_bgr: np.ndarray, face: DetectedFace, output_size: int) -> tuple[np.ndarray, str]:
    if face.has_alignment_landmarks:
        try:
            from uniface.face_utils import face_alignment
        except ImportError as exc:
            raise RuntimeError("UniFace face_alignment is required for landmark-based alignment") from exc
        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        aligned, _ = face_alignment(image_bgr, landmarks, image_size=output_size)
        return aligned, "landmark_5pt"

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    left = int(max(0.0, x1))
    top = int(max(0.0, y1))
    right = int(min(float(width), x2))
    bottom = int(min(float(height), y2))
    if right - left < 2 or bottom - top < 2:
        raise ValueError(f"invalid face bbox after clamping: {face.bbox}")
    return image_bgr[top:bottom, left:right], "bbox"


def predict_image(
    image_path: Path,
    detector: FaceDetector,
    predictor: FastFaceOnnxPredictor,
    max_faces: int,
    selection_metric: str,
) -> dict[str, Any]:
    expanded_image_path = expand_path(image_path)
    image = cv2.imread(str(expanded_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {expanded_image_path}")

    faces = detector.detect(image, max_faces=max_faces, selection_metric=selection_metric)
    result: dict[str, Any] = {
        "status": "no_face" if not faces else "ok",
        "image": str(expanded_image_path),
        "detector": detector.name,
        "fastface_model": str(predictor.model_path),
        "fastface_input_size": predictor.input_size,
        "face_count": len(faces),
        "faces": [],
    }
    if not faces:
        return result

    for index, face in enumerate(faces):
        crop, crop_mode = align_face(image, face, predictor.input_size)
        prediction = predictor.predict(crop)
        result["faces"].append(
            {
                "index": index,
                "bbox": [float(value) for value in face.bbox],
                "detector_score": face.score,
                "landmarks": [[float(x), float(y)] for x, y in face.landmarks],
                "crop_mode": crop_mode,
                "gender": prediction.gender,
                "gender_name": prediction.gender_name,
                "female_prob": prediction.female_prob,
                "male_prob": prediction.male_prob,
                "gender_confidence": prediction.gender_confidence,
                "age": prediction.age,
            }
        )
    return result
