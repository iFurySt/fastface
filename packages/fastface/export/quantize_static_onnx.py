from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from PIL import Image, ImageFile

from fastface.data.manifest_dataset import crop_face
from fastface.paths import expand_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ManifestCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        manifest_paths: list[Path],
        input_name: str,
        input_size: int,
        samples: int,
        batch_size: int,
        face_crop_margin: float,
    ) -> None:
        self.input_name = input_name
        self.input_size = input_size
        self.batch_size = batch_size
        image_items: list[dict] = []
        for manifest_path in [expand_path(path) for path in manifest_paths]:
            with manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    if item.get("split") == "train":
                        image_items.append(item)
                    if len(image_items) >= samples:
                        break
            if len(image_items) >= samples:
                break
        self.face_crop_margin = face_crop_margin
        self._batches = self._make_batches(image_items[:samples])

    def _preprocess(self, item: dict) -> np.ndarray:
        image = Image.open(item["image_path"]).convert("RGB")
        image = crop_face(image, item, margin=self.face_crop_margin).resize((self.input_size, self.input_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))

    def _make_batches(self, image_items: list[dict]) -> Iterator[dict[str, np.ndarray]]:
        for start in range(0, len(image_items), self.batch_size):
            batch_items = image_items[start : start + self.batch_size]
            batch = np.stack([self._preprocess(item) for item in batch_items]).astype(np.float32)
            yield {self.input_name: batch}

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Static-quantize FastFace ONNX with manifest calibration data.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--input-name", default="image")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--face-crop-margin", type=float, default=0.0)
    parser.add_argument("--quant-format", choices=["QDQ", "QOperator"], default="QDQ")
    parser.add_argument("--activation-type", choices=["QInt8", "QUInt8"], default="QUInt8")
    parser.add_argument("--weight-type", choices=["QInt8", "QUInt8"], default="QInt8")
    parser.add_argument("--calibrate-method", choices=["MinMax", "Entropy", "Percentile", "Distribution"], default="MinMax")
    parser.add_argument("--per-channel", dest="per_channel", action="store_true", default=True)
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--reduce-range", action="store_true")
    args = parser.parse_args()

    reader = ManifestCalibrationReader(
        manifest_paths=args.manifest,
        input_name=args.input_name,
        input_size=args.input_size,
        samples=args.samples,
        batch_size=args.batch_size,
        face_crop_margin=args.face_crop_margin,
    )
    quantize_static(
        model_input=str(args.model),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=getattr(QuantFormat, args.quant_format),
        activation_type=getattr(QuantType, args.activation_type),
        weight_type=getattr(QuantType, args.weight_type),
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        calibrate_method=getattr(CalibrationMethod, args.calibrate_method),
    )
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
