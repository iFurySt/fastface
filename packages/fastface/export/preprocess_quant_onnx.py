from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import quant_pre_process


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-process an ONNX graph before ONNX Runtime quantization.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--skip-onnx-shape", action="store_true")
    parser.add_argument("--skip-symbolic-shape", action="store_true")
    parser.add_argument("--auto-merge", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(
        input_model=args.model,
        output_model_path=args.output,
        skip_optimization=args.skip_optimization,
        skip_onnx_shape=args.skip_onnx_shape,
        skip_symbolic_shape=args.skip_symbolic_shape,
        auto_merge=args.auto_merge,
    )
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    print(args.output)


if __name__ == "__main__":
    main()
