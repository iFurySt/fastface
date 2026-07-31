from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from fastface.models.factory import build_age_gender_model
from fastface.paths import expand_path


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(image)
        return output["gender_logits"], output["age_logits"], output["age"]


def load_model(checkpoint_path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(expand_path(checkpoint_path), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_cfg = config.get("model", {})
    model = build_age_gender_model(model_cfg, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def export_fp32(checkpoint_path: Path, output_path: Path, input_size: int, opset: int) -> None:
    model, config = load_model(checkpoint_path)
    wrapped = ExportWrapper(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["gender_logits", "age_logits", "age"],
        dynamic_axes={
            "image": {0: "batch"},
            "gender_logits": {0: "batch"},
            "age_logits": {0: "batch"},
            "age": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "input_size": input_size,
        "opset": opset,
        "outputs": ["gender_logits", "age_logits", "age"],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "config": config,
    }
    with output_path.with_suffix(".metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def export_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    onnx_model = onnx.load(int8_path)
    onnx.checker.check_model(onnx_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FastFace checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--int8-output", type=Path)
    args = parser.parse_args()

    export_fp32(args.checkpoint, args.output, input_size=args.input_size, opset=args.opset)
    if args.int8_output is not None:
        export_int8(args.output, args.int8_output)


if __name__ == "__main__":
    main()
