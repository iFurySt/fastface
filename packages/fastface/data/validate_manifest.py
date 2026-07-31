from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile

from fastface.data.manifest_dataset import parse_bbox

ImageFile.LOAD_TRUNCATED_IMAGES = True

REQUIRED_FIELDS = {
    "sample_id",
    "dataset",
    "split",
    "image_path",
    "gender",
    "age",
    "age_min",
    "age_max",
    "age_label_type",
    "age_loss_weight",
}


def add_error(errors: list[dict[str, Any]], line_number: int, code: str, detail: str, max_errors: int) -> None:
    if len(errors) < max_errors:
        errors.append({"line": line_number, "code": code, "detail": detail})


def as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_manifest(
    manifest_path: Path,
    check_images: bool,
    check_readable: bool,
    max_readable: int,
    max_errors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats: dict[str, Any] = {
        "manifest": str(manifest_path),
        "rows": 0,
        "splits": Counter(),
        "datasets": Counter(),
        "genders": Counter(),
        "age_label_types": Counter(),
        "missing_images": 0,
        "readable_images_checked": 0,
        "unreadable_images": 0,
        "duplicate_sample_ids": 0,
        "bbox_face_rows": 0,
        "invalid_bbox_face_rows": 0,
    }
    errors: list[dict[str, Any]] = []
    sample_ids: set[str] = set()

    if not manifest_path.exists():
        add_error(errors, 0, "missing_manifest", str(manifest_path), max_errors)
        return stats, errors
    if not manifest_path.is_file():
        add_error(errors, 0, "manifest_not_file", str(manifest_path), max_errors)
        return stats, errors

    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                add_error(errors, line_number, "invalid_json", str(exc), max_errors)
                continue

            stats["rows"] += 1
            missing_fields = sorted(REQUIRED_FIELDS - set(item))
            if missing_fields:
                add_error(errors, line_number, "missing_fields", ",".join(missing_fields), max_errors)

            sample_id = str(item.get("sample_id", ""))
            if sample_id in sample_ids:
                stats["duplicate_sample_ids"] += 1
                add_error(errors, line_number, "duplicate_sample_id", sample_id, max_errors)
            elif sample_id:
                sample_ids.add(sample_id)

            split = str(item.get("split", ""))
            dataset = str(item.get("dataset", ""))
            gender = as_int(item.get("gender"))
            age = as_float(item.get("age"))
            age_min = as_int(item.get("age_min"))
            age_max = as_int(item.get("age_max"))
            age_loss_weight = as_float(item.get("age_loss_weight"))
            age_label_type = str(item.get("age_label_type", ""))

            stats["splits"][split] += 1
            stats["datasets"][dataset] += 1
            stats["genders"][str(item.get("gender", ""))] += 1
            stats["age_label_types"][age_label_type] += 1

            if gender not in {0, 1}:
                add_error(errors, line_number, "invalid_gender", str(item.get("gender")), max_errors)
            if age is None or age < 0.0 or age > 100.0:
                add_error(errors, line_number, "invalid_age", str(item.get("age")), max_errors)
            if age_min is None or age_max is None or age_min < 0 or age_max > 100 or age_min > age_max:
                add_error(errors, line_number, "invalid_age_range", f"{item.get('age_min')}..{item.get('age_max')}", max_errors)
            elif age is not None and not (float(age_min) <= age <= float(age_max)):
                add_error(errors, line_number, "age_outside_range", f"age={age} range={age_min}..{age_max}", max_errors)
            if age_label_type not in {"exact", "range"}:
                add_error(errors, line_number, "invalid_age_label_type", age_label_type, max_errors)
            if age_loss_weight is None or age_loss_weight < 0.0:
                add_error(errors, line_number, "invalid_age_loss_weight", str(item.get("age_loss_weight")), max_errors)

            if "bbox_face" in item:
                stats["bbox_face_rows"] += 1
                if parse_bbox(item.get("bbox_face")) is None:
                    stats["invalid_bbox_face_rows"] += 1
                    add_error(errors, line_number, "invalid_bbox_face", str(item.get("bbox_face")), max_errors)

            image_path = Path(str(item.get("image_path", "")))
            if check_images and not image_path.exists():
                stats["missing_images"] += 1
                add_error(errors, line_number, "missing_image", str(image_path), max_errors)
                continue

            should_check_readable = check_readable and (max_readable <= 0 or stats["readable_images_checked"] < max_readable)
            if should_check_readable:
                stats["readable_images_checked"] += 1
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as exc:  # noqa: BLE001 - validation should report any PIL/IO failure.
                    stats["unreadable_images"] += 1
                    add_error(errors, line_number, "unreadable_image", f"{image_path}: {exc}", max_errors)

    stats["splits"] = dict(stats["splits"])
    stats["datasets"] = dict(stats["datasets"])
    stats["genders"] = dict(stats["genders"])
    stats["age_label_types"] = dict(stats["age_label_types"])
    return stats, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FastFace JSONL manifests.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--require-min-rows", type=int, default=0)
    parser.add_argument("--require-split", action="append", default=[])
    parser.add_argument("--require-dataset", action="append", default=[])
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--check-readable", action="store_true")
    parser.add_argument("--max-readable", type=int, default=0, help="0 means all readable checks when --check-readable is set.")
    parser.add_argument("--max-errors", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = []
    errors: list[dict[str, Any]] = []
    total_rows = 0
    total_splits: Counter[str] = Counter()
    total_datasets: Counter[str] = Counter()

    for manifest_path in args.manifest:
        stats, manifest_errors = validate_manifest(
            manifest_path=manifest_path,
            check_images=args.check_images,
            check_readable=args.check_readable,
            max_readable=args.max_readable,
            max_errors=args.max_errors,
        )
        summaries.append(stats)
        errors.extend({"manifest": stats["manifest"], **error} for error in manifest_errors[: args.max_errors])
        total_rows += int(stats["rows"])
        total_splits.update(stats["splits"])
        total_datasets.update(stats["datasets"])

    if total_rows < args.require_min_rows:
        errors.append({"line": 0, "code": "too_few_rows", "detail": f"{total_rows} < {args.require_min_rows}"})
    for split in args.require_split:
        if total_splits.get(split, 0) <= 0:
            errors.append({"line": 0, "code": "missing_required_split", "detail": split})
    for dataset in args.require_dataset:
        if total_datasets.get(dataset, 0) <= 0:
            errors.append({"line": 0, "code": "missing_required_dataset", "detail": dataset})

    report = {
        "ok": not errors,
        "rows": total_rows,
        "splits": dict(total_splits),
        "datasets": dict(total_datasets),
        "manifests": summaries,
        "errors": errors[: args.max_errors],
        "error_count_capped": len(errors) > args.max_errors,
    }
    text = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
