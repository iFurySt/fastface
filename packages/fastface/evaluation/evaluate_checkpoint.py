from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastface.data.manifest_dataset import ManifestDataset
from fastface.models.factory import build_age_gender_model
from fastface.paths import expand_path


def load_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(expand_path(checkpoint_path), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_cfg = config.get("model", {})
    model = build_age_gender_model(model_cfg, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def update_stats(stats: dict[str, float], gender: torch.Tensor, pred_gender: torch.Tensor, age: torch.Tensor, pred_age: torch.Tensor) -> None:
    abs_error = (pred_age - age).abs()
    stats["count"] += float(gender.numel())
    stats["gender_correct"] += float((pred_gender == gender).sum().item())
    stats["age_abs_error"] += float(abs_error.sum().item())
    stats["age_cs5"] += float((abs_error <= 5.0).sum().item())
    for cls, name in ((0, "female"), (1, "male")):
        mask = gender == cls
        stats[f"{name}_count"] += float(mask.sum().item())
        stats[f"{name}_correct"] += float(((pred_gender == gender) & mask).sum().item())


def finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    count = max(stats["count"], 1.0)
    female_count = max(stats["female_count"], 1.0)
    male_count = max(stats["male_count"], 1.0)
    female_acc = stats["female_correct"] / female_count
    male_acc = stats["male_correct"] / male_count
    return {
        "count": int(stats["count"]),
        "gender_acc": stats["gender_correct"] / count,
        "gender_balanced_acc": (female_acc + male_acc) / 2.0,
        "female_acc": female_acc,
        "male_acc": male_acc,
        "age_mae": stats["age_abs_error"] / count,
        "age_cs5": stats["age_cs5"] / count,
    }


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    manifest_paths: list[Path] | None,
    split: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    output_path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(checkpoint_path)
    model.to(device)
    if manifest_paths is None:
        data_cfg = config.get("data", {})
        manifest_paths = [expand_path(path) for path in data_cfg.get("eval_manifests", data_cfg.get("manifests", []))]
    if not manifest_paths:
        raise ValueError("No manifests provided and checkpoint config does not contain data.manifests")
    dataset = ManifestDataset(
        manifest_paths=[expand_path(path) for path in manifest_paths],
        split=split,
        input_size=input_size,
        training=False,
        age_sigma=float(config.get("data", {}).get("age_sigma", 2.0)),
        face_crop_margin=float(config.get("data", {}).get("face_crop_margin", 0.0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    aggregate: dict[str, float] = defaultdict(float)
    by_dataset: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for batch in tqdm(loader, desc="evaluate"):
        image = batch["image"].to(device, non_blocking=True)
        gender = batch["gender"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)
        output = model(image)
        pred_gender = output["gender_logits"].argmax(dim=1)
        pred_age = output["age"]
        update_stats(aggregate, gender.cpu(), pred_gender.cpu(), age.cpu(), pred_age.cpu())
        datasets = batch["dataset"]
        for source in sorted(set(datasets)):
            indices = [idx for idx, name in enumerate(datasets) if name == source]
            tensor_indices = torch.tensor(indices, dtype=torch.long)
            update_stats(
                by_dataset[source],
                gender.cpu().index_select(0, tensor_indices),
                pred_gender.cpu().index_select(0, tensor_indices),
                age.cpu().index_select(0, tensor_indices),
                pred_age.cpu().index_select(0, tensor_indices),
            )
    result = {
        "checkpoint": str(checkpoint_path),
        "manifests": [str(path) for path in manifest_paths],
        "split": split,
        "input_size": input_size,
        "aggregate": finalize_stats(aggregate),
        "by_dataset": {source: finalize_stats(stats) for source, stats in sorted(by_dataset.items())},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FastFace checkpoint on manifest val split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--split", default="val")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate(
        checkpoint_path=args.checkpoint,
        manifest_paths=args.manifest,
        split=args.split,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_path=args.output,
    )
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
