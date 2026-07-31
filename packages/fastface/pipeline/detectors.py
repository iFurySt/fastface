from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


def build_detector(
    backend: str,
    model_name: str | None,
    confidence_threshold: float,
    nms_threshold: float,
    input_size: int,
    providers: list[str] | None,
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
    raise ValueError(f"unsupported detector backend: {backend}")
