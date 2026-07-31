from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from fastface.data.build_manifest import IMAGE_EXTENSIONS


IMDB_CLEAN_TAR_SIZES = {
    "imdb_0.tar": 28_708_782_080,
    "imdb_1.tar": 27_734_599_680,
    "imdb_2.tar": 29_475_174_400,
    "imdb_3.tar": 30_881_392_640,
    "imdb_4.tar": 27_863_429_120,
    "imdb_5.tar": 30_502_092_800,
    "imdb_6.tar": 28_542_679_040,
    "imdb_7.tar": 28_599_234_560,
    "imdb_8.tar": 27_647_651_840,
    "imdb_9.tar": 25_642_557_440,
}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        stat = path.stat()
        return getattr(stat, "st_blocks", 0) * 512 or stat.st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            total += getattr(stat, "st_blocks", 0) * 512 or stat.st_size
    return total


def file_bytes(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    stat = path.stat()
    return getattr(stat, "st_blocks", 0) * 512 or stat.st_size


def progress(actual: int, expected: int) -> dict[str, float | int]:
    return {
        "actual_bytes": actual,
        "expected_bytes": expected,
        "fraction": actual / expected if expected else 0.0,
        "percent": (actual / expected * 100.0) if expected else 0.0,
    }


def imdb_clean_tar_progress(tar_dir: Path) -> dict[str, dict[str, float | int]]:
    return {
        name: progress(file_bytes(tar_dir / name), expected)
        for name, expected in sorted(IMDB_CLEAN_TAR_SIZES.items())
    }


def read_previous_report(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def transfer_rate(actual: int, expected: int, previous_actual: int | None, previous_time: float | None, now: float) -> dict[str, float | int | None] | None:
    if previous_actual is None or previous_time is None:
        return None
    elapsed = now - previous_time
    delta = actual - previous_actual
    if elapsed <= 0 or delta < 0:
        return None
    bytes_per_second = delta / elapsed
    remaining = max(expected - actual, 0)
    return {
        "elapsed_seconds": elapsed,
        "byte_delta": delta,
        "bytes_per_second": bytes_per_second,
        "remaining_bytes": remaining,
        "eta_seconds": remaining / bytes_per_second if bytes_per_second > 0 else None,
    }


def progress_snapshot(name: str, item: dict[str, float | int]) -> dict[str, float | int | str]:
    return {
        "name": name,
        "actual_bytes": int(item["actual_bytes"]),
        "expected_bytes": int(item["expected_bytes"]),
        "fraction": float(item["fraction"]),
        "percent": float(item["percent"]),
    }


def rate_snapshot(name: str, rate: dict[str, float | int | None]) -> dict[str, float | int | str | None]:
    return {
        "name": name,
        "bytes_per_second": rate.get("bytes_per_second"),
        "remaining_bytes": rate.get("remaining_bytes"),
        "eta_seconds": rate.get("eta_seconds"),
    }


def summarize_imdb_tar_progress(
    progress_details: dict[str, dict[str, float | int]],
    rate_details: dict[str, dict[str, float | int | None]],
    total_rate: dict[str, float | int | None] | None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "total_count": len(progress_details),
        "completed_count": sum(1 for item in progress_details.values() if float(item["fraction"]) >= 1.0),
        "active_count": sum(1 for item in progress_details.values() if 0 < int(item["actual_bytes"]) < int(item["expected_bytes"])),
        "pending_count": sum(1 for item in progress_details.values() if int(item["actual_bytes"]) <= 0),
    }
    if progress_details:
        closest_name, closest_item = max(progress_details.items(), key=lambda entry: float(entry[1]["percent"]))
        least_name, least_item = min(progress_details.items(), key=lambda entry: float(entry[1]["percent"]))
        summary["closest_to_complete"] = progress_snapshot(closest_name, closest_item)
        summary["least_complete"] = progress_snapshot(least_name, least_item)

    if total_rate is not None:
        summary["total_bytes_per_second"] = total_rate.get("bytes_per_second")
        summary["total_eta_seconds"] = total_rate.get("eta_seconds")

    eta_candidates = [
        (name, rate)
        for name, rate in rate_details.items()
        if isinstance(rate.get("eta_seconds"), (int, float))
    ]
    if eta_candidates:
        name, rate = min(eta_candidates, key=lambda entry: float(entry[1]["eta_seconds"]))
        summary["earliest_tar_eta"] = rate_snapshot(name, rate)

    active_rate_candidates = [
        (name, rate)
        for name, rate in rate_details.items()
        if isinstance(rate.get("bytes_per_second"), (int, float)) and float(rate["bytes_per_second"]) > 0
    ]
    if active_rate_candidates:
        fastest_name, fastest_rate = max(active_rate_candidates, key=lambda entry: float(entry[1]["bytes_per_second"]))
        slowest_name, slowest_rate = min(active_rate_candidates, key=lambda entry: float(entry[1]["bytes_per_second"]))
        summary["fastest_active_tar"] = rate_snapshot(fastest_name, fastest_rate)
        summary["slowest_active_tar"] = rate_snapshot(slowest_name, slowest_rate)

    return summary


def nested_get(data: dict | None, keys: list[str]) -> object | None:
    value: object = data or {}
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Report FastFace dataset staging status.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    now = time.time()
    previous_report = read_previous_report(args.output)
    previous_time = nested_get(previous_report, ["generated_at_unix"])
    if not isinstance(previous_time, (int, float)):
        previous_time = None

    raw = args.data_root / "raw"
    manifests = args.data_root / "manifests"
    imdb_clean_tar_dir = raw / "imdb_clean" / "tars"
    imdb_clean_tars_bytes = count_bytes(imdb_clean_tar_dir)
    imdb_clean_tars_expected = sum(IMDB_CLEAN_TAR_SIZES.values())
    imdb_clean_tars_progress = imdb_clean_tar_progress(imdb_clean_tar_dir)
    previous_imdb_clean_tars_bytes = nested_get(previous_report, ["raw_bytes", "imdb_clean_tars"])
    if not isinstance(previous_imdb_clean_tars_bytes, int):
        previous_imdb_clean_tars_bytes = None
    imdb_clean_tars_rate = transfer_rate(
        imdb_clean_tars_bytes,
        imdb_clean_tars_expected,
        previous_imdb_clean_tars_bytes,
        previous_time,
        now,
    )
    imdb_clean_tars_rate_details = {}
    for name, item in imdb_clean_tars_progress.items():
        previous_actual = nested_get(previous_report, ["raw_progress_details", "imdb_clean_tars", name, "actual_bytes"])
        if not isinstance(previous_actual, int):
            previous_actual = None
        rate = transfer_rate(
            int(item["actual_bytes"]),
            int(item["expected_bytes"]),
            previous_actual,
            previous_time,
            now,
        )
        if rate is not None:
            imdb_clean_tars_rate_details[name] = rate

    report = {
        "generated_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "generated_at_unix": now,
        "data_root": str(args.data_root),
        "raw_images": {
            "fairface": count_images(raw / "fairface"),
            "utkface": count_images(raw / "utkface"),
            "lagenda_hf_uaebn": count_images(raw / "lagenda_hf_uaebn"),
            "imdb_clean": count_images(raw / "imdb_clean"),
        },
        "manifest_rows": {
            "fairface": count_lines(manifests / "fairface.jsonl"),
            "utkface": count_lines(manifests / "utkface.jsonl"),
            "lagenda_hf_uaebn": count_lines(manifests / "lagenda_hf_uaebn.jsonl"),
            "imdb_clean": count_lines(manifests / "imdb_clean.jsonl"),
        },
        "raw_bytes": {
            "imdb_clean_tars": imdb_clean_tars_bytes,
        },
        "raw_bytes_expected": {
            "imdb_clean_tars": imdb_clean_tars_expected,
        },
        "raw_progress": {
            "imdb_clean_tars": progress(imdb_clean_tars_bytes, imdb_clean_tars_expected),
        },
        "raw_progress_details": {
            "imdb_clean_tars": imdb_clean_tars_progress,
        },
        "raw_progress_summary": {
            "imdb_clean_tars": summarize_imdb_tar_progress(
                imdb_clean_tars_progress,
                imdb_clean_tars_rate_details,
                imdb_clean_tars_rate,
            ),
        },
    }
    if imdb_clean_tars_rate is not None:
        report["raw_rate_since_previous"] = {
            "imdb_clean_tars": imdb_clean_tars_rate,
        }
    if imdb_clean_tars_rate_details:
        report["raw_rate_details_since_previous"] = {
            "imdb_clean_tars": imdb_clean_tars_rate_details,
        }
    text = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
