from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

from fastface.data.labels import exact_age, fairface_age, normalize_gender, normalize_utk_gender


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".csv") or name.startswith("__MACOSX/"):
                    continue
                with archive.open(name) as raw:
                    text = (line.decode("utf-8-sig") for line in raw)
                    for row in csv.DictReader(text):
                        row["__source_file"] = name
                        yield row
        return

    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            row["__source_file"] = path.name
            yield row


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def resolve_annotation_image_path(images_dir: Path, rel_path: str) -> Path:
    image_path = images_dir / rel_path
    if image_path.exists():
        return image_path
    nested_path = images_dir / "imdb" / rel_path
    if nested_path.exists():
        return nested_path
    fallback_path = images_dir / Path(rel_path).name
    if fallback_path.exists():
        return fallback_path
    return image_path


def fairface_rows(raw_dir: Path, require_images: bool) -> Iterator[dict]:
    for split, csv_name in (("train", "fairface_label_train.csv"), ("val", "fairface_label_val.csv")):
        csv_path = raw_dir / csv_name
        for index, row in enumerate(read_csv_rows(csv_path)):
            gender = normalize_gender(row.get("gender"))
            age = fairface_age(row.get("age"))
            if gender is None or age is None:
                continue
            image_path = raw_dir / row["file"]
            if require_images and not image_path.exists():
                continue
            yield {
                "sample_id": f"fairface:{split}:{index}",
                "dataset": "fairface",
                "split": split,
                "image_path": str(image_path),
                "gender": gender,
                "gender_original": row.get("gender", ""),
                "age": age.age,
                "age_min": age.age_min,
                "age_max": age.age_max,
                "age_label_type": age.label_type,
                "age_loss_weight": age.loss_weight,
                "source_label": row.get("age", ""),
            }


def lagenda_rows(annotation_path: Path, images_dir: Path, split_seed: int, require_images: bool) -> Iterator[dict]:
    rng = random.Random(split_seed)
    for index, row in enumerate(read_csv_rows(annotation_path)):
        gender = normalize_gender(row.get("gender"))
        age = exact_age(row.get("age"))
        if gender is None or age is None:
            continue
        rel_path = row.get("img_name", "")
        image_path = resolve_annotation_image_path(images_dir, rel_path)
        if require_images and not image_path.exists():
            continue
        split = "val" if rng.random() < 0.1 else "train"
        yield {
            "sample_id": f"lagenda:{index}",
            "dataset": "lagenda",
            "split": split,
            "image_path": str(image_path),
            "gender": gender,
            "gender_original": row.get("gender", ""),
            "age": age.age,
            "age_min": age.age_min,
            "age_max": age.age_max,
            "age_label_type": age.label_type,
            "age_loss_weight": age.loss_weight,
            "bbox_face": [row.get("face_x0"), row.get("face_y0"), row.get("face_x1"), row.get("face_y1")],
            "bbox_person": [row.get("person_x0"), row.get("person_y0"), row.get("person_x1"), row.get("person_y1")],
        }


def imdb_clean_rows(annotation_path: Path, images_dir: Path, require_images: bool) -> Iterator[dict]:
    for split_name in ("train", "valid", "test"):
        split = "val" if split_name in {"valid", "test"} else "train"
        for index, row in enumerate(read_csv_rows(annotation_path)):
            source_name = row.get("__source_file", "")
            if source_name and split_name not in source_name:
                continue
            gender = normalize_gender(row.get("gender"))
            age = exact_age(row.get("age"))
            if gender is None or age is None:
                continue
            rel_path = row.get("img_name") or row.get("filename") or ""
            image_path = resolve_annotation_image_path(images_dir, rel_path)
            if require_images and not image_path.exists():
                continue
            yield {
                "sample_id": f"imdb-clean:{split_name}:{index}",
                "dataset": "imdb-clean",
                "split": split,
                "image_path": str(image_path),
                "gender": gender,
                "gender_original": row.get("gender", ""),
                "age": age.age,
                "age_min": age.age_min,
                "age_max": age.age_max,
                "age_label_type": age.label_type,
                "age_loss_weight": age.loss_weight,
                "bbox_face": [row.get("face_x0"), row.get("face_y0"), row.get("face_x1"), row.get("face_y1")],
                "bbox_person": [row.get("person_x0"), row.get("person_y0"), row.get("person_x1"), row.get("person_y1")],
            }


def utkface_rows(raw_dir: Path, split_seed: int, require_images: bool) -> Iterator[dict]:
    rng = random.Random(split_seed)
    files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    for index, image_path in enumerate(sorted(files)):
        parts = image_path.name.split("_")
        if len(parts) < 4:
            continue
        age = exact_age(parts[0])
        gender = normalize_utk_gender(parts[1])
        if gender is None or age is None:
            continue
        if require_images and not image_path.exists():
            continue
        split = "val" if rng.random() < 0.1 else "train"
        yield {
            "sample_id": f"utkface:{index}",
            "dataset": "utkface",
            "split": split,
            "image_path": str(image_path),
            "gender": gender,
            "gender_original": parts[1],
            "age": age.age,
            "age_min": age.age_min,
            "age_max": age.age_max,
            "age_label_type": age.label_type,
            "age_loss_weight": age.loss_weight,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FastFace JSONL manifests.")
    parser.add_argument("--dataset", choices=["fairface", "lagenda", "imdb-clean", "utkface"], required=True)
    parser.add_argument("--raw-dir", type=Path, help="Dataset raw directory.")
    parser.add_argument("--annotation", type=Path, help="Annotation CSV or ZIP.")
    parser.add_argument("--images-dir", type=Path, help="Image root for annotation-driven datasets.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_images = not args.allow_missing_images
    if args.dataset == "fairface":
        if args.raw_dir is None:
            raise SystemExit("--raw-dir is required for FairFace")
        rows = fairface_rows(args.raw_dir, require_images=require_images)
    elif args.dataset == "lagenda":
        if args.annotation is None or args.images_dir is None:
            raise SystemExit("--annotation and --images-dir are required for Lagenda")
        rows = lagenda_rows(args.annotation, args.images_dir, args.split_seed, require_images=require_images)
    elif args.dataset == "imdb-clean":
        if args.annotation is None or args.images_dir is None:
            raise SystemExit("--annotation and --images-dir are required for IMDB-clean")
        rows = imdb_clean_rows(args.annotation, args.images_dir, require_images=require_images)
    elif args.dataset == "utkface":
        if args.raw_dir is None:
            raise SystemExit("--raw-dir is required for UTKFace")
        rows = utkface_rows(args.raw_dir, args.split_seed, require_images=require_images)
    else:
        raise AssertionError(args.dataset)

    count = write_jsonl(args.out, rows)
    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
