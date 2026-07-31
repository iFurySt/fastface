from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_run(run_dir: Path) -> Path:
    rows = []
    for path in sorted((run_dir / "cpu-thread-sweep").glob("*.json")):
        payload = json.loads(path.read_text())
        rows.extend(payload["results"])
    summary = {}
    for row in rows:
        key = (Path(row["model"]).stem, row["batch_size"])
        current = summary.get(key)
        if current is None or row["images_per_second"] > current["images_per_second"]:
            summary[key] = row
    out = {
        "run": run_dir.name,
        "best_by_model_and_batch": [
            {"model": model, "batch_size": batch, **metrics}
            for (model, batch), metrics in sorted(summary.items())
        ],
        "all_results": rows,
    }
    out_path = run_dir / "cpu-thread-sweep-summary.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FastFace ONNX CPU thread sweep results.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    for run_dir in args.run_dir:
        print(summarize_run(run_dir))


if __name__ == "__main__":
    main()
