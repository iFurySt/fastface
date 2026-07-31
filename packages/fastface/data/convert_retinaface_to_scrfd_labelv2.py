from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def iter_retinaface_items(label_path: Path):
    current_name: str | None = None
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_name is not None:
                yield current_name, rows
            current_name = line[2:].strip()
            rows = []
            continue
        rows.append([float(value) for value in line.split()])
    if current_name is not None:
        yield current_name, rows


def convert(label_path: Path, image_root: Path, output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "images": 0,
        "faces": 0,
        "missing_images": 0,
        "skipped_faces": 0,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for image_name, rows in iter_retinaface_items(label_path):
            image_path = image_root / image_name
            if not image_path.exists():
                stats["missing_images"] += 1
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            converted_rows: list[list[float]] = []
            for row in rows:
                if len(row) < 19:
                    stats["skipped_faces"] += 1
                    continue
                x, y, box_width, box_height = row[:4]
                if box_width <= 1 or box_height <= 1:
                    stats["skipped_faces"] += 1
                    continue
                converted = [x, y, x + box_width, y + box_height]
                for offset in range(4, 19, 3):
                    point_x = row[offset]
                    point_y = row[offset + 1]
                    visibility = row[offset + 2]
                    converted.extend([point_x, point_y, visibility])
                converted_rows.append(converted)
            if not converted_rows:
                continue
            handle.write(f"# {image_name} {width} {height}\n")
            for converted in converted_rows:
                handle.write(" ".join(f"{value:.3f}" for value in converted) + "\n")
                stats["faces"] += 1
            stats["images"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RetinaFace label.txt to SCRFD labelv2.txt.")
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = convert(label_path=args.label, image_root=args.image_root, output_path=args.output)
    print(stats)


if __name__ == "__main__":
    main()
