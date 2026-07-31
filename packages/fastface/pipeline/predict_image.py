from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastface.pipeline.detectors import build_detector


def parse_providers(value: str) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-image FastFace age/gender inference.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--model", type=Path, required=True, help="FastFace ONNX model path.")
    parser.add_argument(
        "--detector",
        choices=["retinaface", "scrfd", "owned-retinaface-onnx"],
        default="retinaface",
        help="Detector backend. RetinaFace and SCRFD use UniFace baseline implementations.",
    )
    parser.add_argument(
        "--detector-model",
        help=(
            "Optional detector model enum name or value, for example retinaface_mnet_v2, "
            "MNET_V2, scrfd_500m, or SCRFD_10G_KPS."
        ),
    )
    parser.add_argument("--detector-input-size", type=int, default=640)
    parser.add_argument("--detector-resize-mode", choices=["square", "max-side"], default="square")
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-nms", type=float, default=0.4)
    parser.add_argument("--detector-pre-nms-topk", type=int, default=1000)
    parser.add_argument("--detector-post-nms-topk", type=int, default=750)
    parser.add_argument("--owned-detector-onnx", type=Path, help="Owned RetinaFace ONNX detector path.")
    parser.add_argument("--owned-detector-network", default="mobilenetv1_0.50")
    parser.add_argument("--providers", default="CPUExecutionProvider", help="Comma-separated ONNX Runtime providers for detector.")
    parser.add_argument("--input-size", type=int, help="Override FastFace input size. Defaults to ONNX metadata.")
    parser.add_argument("--max-faces", type=int, default=0, help="Maximum faces to return. Use 0 for all faces.")
    parser.add_argument(
        "--selection-metric",
        choices=["max", "default"],
        default="max",
        help="Face ranking metric when --max-faces is set.",
    )
    parser.add_argument("--intra-op-num-threads", type=int, default=0)
    parser.add_argument("--inter-op-num-threads", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    from fastface.pipeline.runtime import FastFaceOnnxPredictor, predict_image

    detector = build_detector(
        backend=args.detector,
        model_name=args.detector_model,
        confidence_threshold=args.detector_conf,
        nms_threshold=args.detector_nms,
        input_size=args.detector_input_size,
        providers=parse_providers(args.providers),
        owned_onnx_path=str(args.owned_detector_onnx) if args.owned_detector_onnx is not None else None,
        owned_network=args.owned_detector_network,
        resize_mode=args.detector_resize_mode,
        pre_nms_topk=args.detector_pre_nms_topk,
        post_nms_topk=args.detector_post_nms_topk,
    )
    predictor = FastFaceOnnxPredictor(
        model_path=args.model,
        input_size=args.input_size,
        intra_op_num_threads=args.intra_op_num_threads,
        inter_op_num_threads=args.inter_op_num_threads,
    )
    result = predict_image(
        image_path=args.image,
        detector=detector,
        predictor=predictor,
        max_faces=args.max_faces,
        selection_metric=args.selection_metric,
    )

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
