from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_session(
    model_path: Path,
    intra_op_num_threads: int,
    inter_op_num_threads: int,
    execution_mode: str,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_num_threads > 0:
        options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads > 0:
        options.inter_op_num_threads = inter_op_num_threads
    if execution_mode == "parallel":
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    elif execution_mode == "sequential":
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def benchmark_model(
    model_path: Path,
    input_size: int,
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
    intra_op_num_threads: int,
    inter_op_num_threads: int,
    execution_mode: str,
) -> dict:
    session = make_session(
        model_path=model_path,
        intra_op_num_threads=intra_op_num_threads,
        inter_op_num_threads=inter_op_num_threads,
        execution_mode=execution_mode,
    )
    input_name = session.get_inputs()[0].name
    results: list[dict] = []
    rng = np.random.default_rng(20260730)
    for batch_size in batch_sizes:
        image = rng.normal(size=(batch_size, 3, input_size, input_size)).astype(np.float32)
        for _ in range(warmup):
            session.run(None, {input_name: image})
        start = time.perf_counter()
        for _ in range(iterations):
            session.run(None, {input_name: image})
        elapsed = time.perf_counter() - start
        latency_ms = elapsed * 1000.0 / iterations
        results.append(
            {
                "model": str(model_path),
                "batch_size": batch_size,
                "iterations": iterations,
                "warmup": warmup,
                "intra_op_num_threads": intra_op_num_threads,
                "inter_op_num_threads": inter_op_num_threads,
                "execution_mode": execution_mode,
                "latency_ms": latency_ms,
                "images_per_second": batch_size * iterations / elapsed,
            }
        )
    return {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime CPU throughput.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--intra-op-num-threads", type=int, default=0)
    parser.add_argument("--inter-op-num-threads", type=int, default=0)
    parser.add_argument("--execution-mode", choices=["sequential", "parallel"], default="sequential")
    args = parser.parse_args()

    result = benchmark_model(
        model_path=args.model,
        input_size=args.input_size,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        iterations=args.iterations,
        intra_op_num_threads=args.intra_op_num_threads,
        inter_op_num_threads=args.inter_op_num_threads,
        execution_mode=args.execution_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
