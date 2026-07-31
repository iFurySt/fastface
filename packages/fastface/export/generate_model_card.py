from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_float(value: Any, digits: int = 5) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return "-"


def fmt_size(path: Path) -> str:
    if not path.exists():
        return "-"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return "-"


def best_by_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get(metric, float("-inf"))))


def load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.resolved.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def model_label(model_cfg: dict[str, Any]) -> str:
    name = str(model_cfg.get("name", "mobilenetv3_age_gender"))
    if name == "mobilenetv3_age_gender":
        return f"MobileNetV3-{str(model_cfg.get('variant', '-')).title()}"
    if name == "torchvision_age_gender":
        backbone = str(model_cfg.get("backbone", "-"))
        return {
            "convnext_tiny": "ConvNeXt-Tiny",
            "efficientnet_b0": "EfficientNet-B0",
            "efficientnet_v2_s": "EfficientNetV2-S",
            "resnet18": "ResNet18",
            "swin_t": "Swin-T",
        }.get(backbone, backbone)
    return name


def append_default_benchmark(lines: list[str], run_dir: Path, path: Path, title: str) -> None:
    data = read_json(path)
    if not data:
        return
    rows = data.get("results", []) if isinstance(data, dict) else data
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Batch | Images/s | Latency ms | Intra Threads | Inter Threads |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            f"{row.get('batch_size', '-')} | "
            f"{fmt_float(row.get('images_per_second'), 1)} | "
            f"{fmt_float(row.get('latency_ms'), 3)} | "
            f"{row.get('intra_op_num_threads', '-')} | "
            f"{row.get('inter_op_num_threads', '-')} |"
        )
    lines.append("")


def append_thread_sweep(lines: list[str], path: Path) -> None:
    data = read_json(path)
    if not data:
        return
    lines.append("## Best Thread Sweep")
    lines.append("")
    lines.append("| Model | Batch | Images/s | Latency ms | Intra Threads | Inter Threads |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in data.get("best_by_model_and_batch", []):
        model = Path(str(row.get("model", ""))).name or "-"
        lines.append(
            "| "
            f"`{model}` | "
            f"{row.get('batch_size', '-')} | "
            f"{fmt_float(row.get('images_per_second'), 1)} | "
            f"{fmt_float(row.get('latency_ms'), 3)} | "
            f"{row.get('intra_op_num_threads', '-')} | "
            f"{row.get('inter_op_num_threads', '-')} |"
        )
    lines.append("")


def build_card(run_dir: Path) -> str:
    run_name = run_dir.name
    config = load_config(run_dir)
    metrics_rows = read_metrics(run_dir / "metrics.jsonl")
    best = best_by_metric(metrics_rows, "val_gender_balanced_acc") or {}
    evaluation = read_json(run_dir / "evaluation_val.json") or {}
    aggregate = evaluation.get("aggregate", {})
    by_dataset = evaluation.get("by_dataset", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    distill_cfg = config.get("distillation", {})

    lines = [
        f"# {run_name}",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Backbone | {model_label(model_cfg)} |",
        f"| Input size | {data_cfg.get('input_size', '-')} |",
        f"| Face crop margin | {data_cfg.get('face_crop_margin', 0.0)} |",
        f"| Epochs | {config.get('train', {}).get('epochs', '-')} |",
        f"| Best epoch | {best.get('epoch', '-')} |",
        f"| Distillation | {'yes' if distill_cfg.get('enabled') else 'no'} |",
        f"| FP32 ONNX | {fmt_size(run_dir / 'model_fp32.onnx')} |",
        f"| Static INT8 ONNX | {fmt_size(run_dir / 'model_int8_static.onnx')} |",
        "",
        "## Validation",
        "",
        "| Metric | Training Best | Evaluation File |",
        "| --- | ---: | ---: |",
        f"| Gender balanced acc | {fmt_float(best.get('val_gender_balanced_acc'))} | {fmt_float(aggregate.get('gender_balanced_acc'))} |",
        f"| Gender acc | {fmt_float(best.get('val_gender_acc'))} | {fmt_float(aggregate.get('gender_acc'))} |",
        f"| Female acc | {fmt_float(best.get('val_female_acc'))} | {fmt_float(aggregate.get('female_acc'))} |",
        f"| Male acc | {fmt_float(best.get('val_male_acc'))} | {fmt_float(aggregate.get('male_acc'))} |",
        f"| Age MAE | {fmt_float(best.get('val_age_mae'))} | {fmt_float(aggregate.get('age_mae'))} |",
        f"| Age CS@5 | {fmt_float(best.get('val_age_cs5'))} | {fmt_float(aggregate.get('age_cs5'))} |",
        "",
    ]

    if by_dataset:
        lines.extend(
            [
                "## Source Slices",
                "",
                "| Source | Count | Gender Balanced Acc | Gender Acc | Age MAE | Age CS@5 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for source, row in sorted(by_dataset.items()):
            lines.append(
                "| "
                f"{source} | "
                f"{row.get('count', '-')} | "
                f"{fmt_float(row.get('gender_balanced_acc'))} | "
                f"{fmt_float(row.get('gender_acc'))} | "
                f"{fmt_float(row.get('age_mae'))} | "
                f"{fmt_float(row.get('age_cs5'))} |"
            )
        lines.append("")

    append_default_benchmark(lines, run_dir, run_dir / "benchmark_fp32_cpu.json", "Default CPU Benchmark: FP32")
    append_default_benchmark(lines, run_dir, run_dir / "benchmark_int8_static_cpu.json", "Default CPU Benchmark: Static INT8")
    append_thread_sweep(lines, run_dir / "cpu-thread-sweep-summary.json")

    artifact_names = [
        "best.pt",
        "last.pt",
        "config.resolved.yaml",
        "metrics.jsonl",
        "evaluation_val.json",
        "model_fp32.onnx",
        "model_int8_static.onnx",
        "benchmark_fp32_cpu.json",
        "benchmark_int8_static_cpu.json",
        "cpu-thread-sweep-summary.json",
    ]
    lines.extend(["## Artifacts", "", "```text", str(run_dir)])
    lines.extend(f"  {name}" for name in artifact_names if (run_dir / name).exists())
    lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown model card for a FastFace run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "model_card.md"
    output.write_text(build_card(args.run_dir), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
