from __future__ import annotations

import argparse
from pathlib import Path


def convert_widerface_bbx_to_retinaface_label(source_path: Path, output_path: Path) -> dict[str, int]:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idx = 0
    written_images = 0
    written_faces = 0
    with output_path.open("w", encoding="utf-8") as handle:
        while idx < len(lines):
            image_name = lines[idx].strip()
            idx += 1
            if not image_name:
                continue
            face_count = int(lines[idx].strip())
            idx += 1
            faces: list[list[float]] = []
            annotation_rows = face_count if face_count > 0 else 1
            for _ in range(annotation_rows):
                parts = [float(value) for value in lines[idx].strip().split()]
                idx += 1
                if face_count == 0:
                    continue
                x, y, width, height = parts[:4]
                if width <= 1 or height <= 1:
                    continue
                points = [
                    (x + 0.35 * width, y + 0.40 * height),
                    (x + 0.65 * width, y + 0.40 * height),
                    (x + 0.50 * width, y + 0.55 * height),
                    (x + 0.38 * width, y + 0.75 * height),
                    (x + 0.62 * width, y + 0.75 * height),
                ]
                row = [x, y, width, height]
                for point_x, point_y in points:
                    row.extend([point_x, point_y, 0.0])
                faces.append(row)
            if not faces:
                continue
            handle.write(f"# {image_name}\n")
            for row in faces:
                handle.write(" ".join(f"{value:.3f}" for value in row) + "\n")
                written_faces += 1
            written_images += 1

    return {
        "images": written_images,
        "faces": written_faces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RetinaFace bootstrap labels from WIDER FACE bbox annotations.")
    parser.add_argument("--source", type=Path, required=True, help="Path to wider_face_train_bbx_gt.txt.")
    parser.add_argument("--output", type=Path, required=True, help="Output RetinaFace-style label.txt path.")
    args = parser.parse_args()

    stats = convert_widerface_bbx_to_retinaface_label(args.source, args.output)
    print(f"wrote {stats['images']} images and {stats['faces']} pseudo-landmark faces to {args.output}")


if __name__ == "__main__":
    main()
