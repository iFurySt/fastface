from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fastface.data.manifest_dataset import crop_face, make_transforms, stable_dataset_seed
from fastface.evaluation.evaluate_checkpoint import load_checkpoint
from fastface.paths import expand_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

GENDER_NAMES = {0: "female", 1: "male"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: Path
    input_size: int
    face_crop_margin: float


def load_manifest_items(
    manifest_paths: list[Path],
    split: str,
    sample_limits: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for manifest_path in [expand_path(path) for path in manifest_paths]:
        with manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("split") == split:
                    items.append(item)
    if not sample_limits:
        return items

    selected_ids: set[str] = set()
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_dataset[str(item.get("dataset", ""))].append(item)
    for dataset, source_items in by_dataset.items():
        limit = sample_limits.get(dataset)
        if limit is not None and limit > 0 and len(source_items) > limit:
            rng = random.Random(stable_dataset_seed(seed, dataset))
            source_items = rng.sample(source_items, limit)
        selected_ids.update(str(item.get("sample_id", "")) for item in source_items)
    return [item for item in items if str(item.get("sample_id", "")) in selected_ids]


class ItemImageDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], input_size: int, face_crop_margin: float) -> None:
        self.items = items
        self.transform = make_transforms(input_size=input_size, training=False)
        self.face_crop_margin = face_crop_margin

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = Image.open(item["image_path"]).convert("RGB")
        image = crop_face(image, item, margin=self.face_crop_margin)
        return {
            "index": index,
            "image": self.transform(image),
        }


def softmax_numpy(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True).clip(min=1e-8)


@torch.no_grad()
def predict_fastface(
    spec: ModelSpec,
    items: list[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    model, _ = load_checkpoint(spec.path)
    model.to(device)
    model.eval()

    dataset = ItemImageDataset(items, input_size=spec.input_size, face_crop_margin=spec.face_crop_margin)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    predictions: dict[str, dict[str, Any]] = {}
    for batch in tqdm(loader, desc=f"predict {spec.name}"):
        image = batch["image"].to(device, non_blocking=True)
        output = model(image)
        probs = torch.softmax(output["gender_logits"], dim=1).cpu().numpy()
        ages = output["age"].detach().cpu().numpy()
        indices = batch["index"].cpu().numpy()
        for row_idx, gender_probs, age_value in zip(indices, probs, ages):
            item = items[int(row_idx)]
            gender = int(np.argmax(gender_probs))
            predictions[str(item["sample_id"])] = {
                "gender": gender,
                "gender_name": GENDER_NAMES[gender],
                "female_prob": float(gender_probs[0]),
                "male_prob": float(gender_probs[1]),
                "gender_confidence": float(np.max(gender_probs)),
                "age": float(age_value),
            }
    return predictions


def load_public_fairface_images(items: list[dict[str, Any]], input_size: int) -> np.ndarray:
    batch: list[np.ndarray] = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for item in items:
        image = Image.open(item["image_path"]).convert("RGB")
        # The public FairFace ONNX model was trained with aligned face crops and uses
        # 0.25 bbox padding when a raw image plus bbox is provided.
        image = crop_face(image, item, margin=0.25)
        array = np.asarray(image.resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        array = (array - mean) / std
        batch.append(np.transpose(array, (2, 0, 1)))
    return np.stack(batch, axis=0).astype(np.float32)


def predict_public_fairface_onnx(
    spec: ModelSpec,
    items: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    session = ort.InferenceSession(str(spec.path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    predictions: dict[str, dict[str, Any]] = {}
    for start in tqdm(range(0, len(items), batch_size), desc=f"predict {spec.name}"):
        batch_items = items[start : start + batch_size]
        batch = load_public_fairface_images(batch_items, input_size=spec.input_size)
        outputs = session.run(output_names, {input_name: batch})
        if len(outputs) < 3:
            raise RuntimeError(f"{spec.name} expected race/gender/age outputs, got {len(outputs)}")
        gender_probs_public = softmax_numpy(np.asarray(outputs[1]), axis=1)
        age_probs = softmax_numpy(np.asarray(outputs[2]), axis=1)
        for item, public_gender_probs, public_age_probs in zip(batch_items, gender_probs_public, age_probs):
            # Public FairFace label order is ["Male", "Female"]; FastFace uses
            # 0=female, 1=male.
            male_prob = float(public_gender_probs[0])
            female_prob = float(public_gender_probs[1])
            gender = 1 if male_prob >= female_prob else 0
            age_group = int(np.argmax(public_age_probs))
            predictions[str(item["sample_id"])] = {
                "gender": gender,
                "gender_name": GENDER_NAMES[gender],
                "female_prob": female_prob,
                "male_prob": male_prob,
                "gender_confidence": max(female_prob, male_prob),
                "age_group": age_group,
            }
    return predictions


def add_optional_repo_to_path(repo_path: Path | None, package_name: str) -> None:
    if package_name in sys.modules:
        return
    if repo_path is None:
        return
    if not repo_path.exists():
        raise FileNotFoundError(f"Optional dependency repo does not exist: {repo_path}")
    sys.path.insert(0, str(repo_path))


def patch_timm_for_mivolo() -> None:
    try:
        from timm.models import _factory
        from timm.models import _helpers
        from timm.models import _pretrained
    except ImportError:
        return
    if hasattr(_helpers, "remap_checkpoint"):
        pass
    else:

        def remap_checkpoint(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return state_dict

        _helpers.remap_checkpoint = remap_checkpoint
    if not hasattr(_pretrained, "split_model_name_tag") and hasattr(_factory, "split_model_name_tag"):
        _pretrained.split_model_name_tag = _factory.split_model_name_tag


def pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


@torch.no_grad()
def predict_mivolo_face(
    spec: ModelSpec,
    items: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    mivolo_repo: Path | None,
) -> dict[str, dict[str, Any]]:
    add_optional_repo_to_path(mivolo_repo, package_name="mivolo")
    patch_timm_for_mivolo()
    try:
        from mivolo.data.misc import prepare_classification_images
        from mivolo.model.create_timm_model import create_model
        from timm.data import resolve_data_config
    except ImportError as exc:
        raise RuntimeError(
            "mivolo_face requires the MiVOLO repository/package and its dependencies. "
            "Set --mivolo-repo or install MiVOLO into the active environment."
        ) from exc

    state = torch.load(expand_path(spec.path), map_location="cpu")
    if state.get("no_gender"):
        raise ValueError(f"{spec.name} checkpoint is age-only and has no gender head")
    with_persons_model = bool(
        state.get("with_persons_model", "patch_embed.conv1.0.weight" in state.get("state_dict", {}))
    )
    input_size = int(state["state_dict"]["pos_embed"].shape[1] * 16)
    if spec.input_size > 0 and input_size != spec.input_size:
        raise ValueError(f"{spec.name} expected input_size={spec.input_size}, checkpoint uses {input_size}")
    model = create_model(
        model_name=f"mivolo_d1_{input_size}",
        num_classes=3,
        in_chans=6 if with_persons_model else 3,
        pretrained=False,
        checkpoint_path=str(expand_path(spec.path)),
        filter_keys=["fds."],
    )
    data_config = resolve_data_config(model=model, verbose=False, use_test_size=True)
    data_config["crop_pct"] = 1.0
    model.to(device)
    model.eval()
    use_half = device.type == "cuda"
    if use_half:
        model.half()

    predictions: dict[str, dict[str, Any]] = {}
    for start in tqdm(range(0, len(items), batch_size), desc=f"predict {spec.name}"):
        batch_items = items[start : start + batch_size]
        crops: list[np.ndarray] = []
        for item in batch_items:
            image = Image.open(item["image_path"]).convert("RGB")
            image = crop_face(image, item, margin=spec.face_crop_margin)
            crops.append(pil_to_bgr_array(image))

        face_input = prepare_classification_images(
            crops,
            input_size,
            data_config["mean"],
            data_config["std"],
            device=device,
        )
        if with_persons_model:
            empty_person_input = prepare_classification_images(
                [None] * len(crops),
                input_size,
                data_config["mean"],
                data_config["std"],
                device=device,
            )
            model_input = torch.cat((face_input, empty_person_input), dim=1)
        else:
            model_input = face_input

        if use_half:
            model_input = model_input.half()
        output = model(model_input)
        gender_probs = torch.softmax(output[:, :2].float(), dim=1).cpu().numpy()
        ages = (
            output[:, 2].float() * (float(state["max_age"]) - float(state["min_age"]))
            + float(state["avg_age"])
        ).cpu().numpy()
        for item, mivolo_gender_probs, age_value in zip(batch_items, gender_probs, ages):
            # MiVOLO gender order is ["male", "female"]. FastFace uses
            # 0=female, 1=male.
            male_prob = float(mivolo_gender_probs[0])
            female_prob = float(mivolo_gender_probs[1])
            gender = 1 if male_prob >= female_prob else 0
            predictions[str(item["sample_id"])] = {
                "gender": gender,
                "gender_name": GENDER_NAMES[gender],
                "female_prob": female_prob,
                "male_prob": male_prob,
                "gender_confidence": max(female_prob, male_prob),
                "age": float(age_value),
            }
    return predictions


def empty_stats() -> dict[str, float]:
    return defaultdict(float)


def update_gender_stats(stats: dict[str, float], label: int, pred: int) -> None:
    stats["count"] += 1.0
    stats["gender_correct"] += float(label == pred)
    for cls, name in ((0, "female"), (1, "male")):
        if label == cls:
            stats[f"{name}_count"] += 1.0
            stats[f"{name}_correct"] += float(pred == label)


def finalize_gender_stats(stats: dict[str, float]) -> dict[str, Any]:
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
    }


def summarize(
    items: list[dict[str, Any]],
    model_predictions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}, "pairwise_disagreements": {}}
    for model_name, predictions in model_predictions.items():
        aggregate = empty_stats()
        by_dataset: dict[str, dict[str, float]] = defaultdict(empty_stats)
        for item in items:
            sample_id = str(item["sample_id"])
            pred = predictions[sample_id]["gender"]
            label = int(item["gender"])
            update_gender_stats(aggregate, label, pred)
            update_gender_stats(by_dataset[str(item.get("dataset", ""))], label, pred)
        summary["models"][model_name] = {
            "aggregate": finalize_gender_stats(aggregate),
            "by_dataset": {name: finalize_gender_stats(stats) for name, stats in sorted(by_dataset.items())},
        }

    names = list(model_predictions)
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            disagreements = 0
            both_correct = 0
            left_only_correct = 0
            right_only_correct = 0
            both_wrong_same = 0
            for item in items:
                sample_id = str(item["sample_id"])
                label = int(item["gender"])
                left_pred = int(model_predictions[left][sample_id]["gender"])
                right_pred = int(model_predictions[right][sample_id]["gender"])
                if left_pred != right_pred:
                    disagreements += 1
                left_correct = left_pred == label
                right_correct = right_pred == label
                both_correct += int(left_correct and right_correct)
                left_only_correct += int(left_correct and not right_correct)
                right_only_correct += int(right_correct and not left_correct)
                both_wrong_same += int((not left_correct) and (not right_correct) and left_pred == right_pred)
            key = f"{left}__vs__{right}"
            summary["pairwise_disagreements"][key] = {
                "count": len(items),
                "disagreements": disagreements,
                "disagreement_rate": disagreements / max(len(items), 1),
                "both_correct": both_correct,
                "left_only_correct": left_only_correct,
                "right_only_correct": right_only_correct,
                "both_wrong_same": both_wrong_same,
            }
    return summary


def build_rows(
    items: list[dict[str, Any]],
    model_predictions: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    model_names = list(model_predictions)
    rows: list[dict[str, Any]] = []
    for item in items:
        sample_id = str(item["sample_id"])
        label = int(item["gender"])
        genders = {name: int(model_predictions[name][sample_id]["gender"]) for name in model_names}
        confidences = {name: float(model_predictions[name][sample_id]["gender_confidence"]) for name in model_names}
        correct = [name for name, gender in genders.items() if gender == label]
        wrong = [name for name, gender in genders.items() if gender != label]
        unique_preds = sorted(set(genders.values()))
        if len(unique_preds) <= 1:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": str(item.get("dataset", "")),
                "image_path": str(item["image_path"]),
                "label_gender": label,
                "label_gender_name": GENDER_NAMES[label],
                "label_age": float(item["age"]),
                "correct_models": "|".join(correct),
                "wrong_models": "|".join(wrong),
                "num_correct_models": len(correct),
                "num_wrong_models": len(wrong),
                "gender_predictions": "|".join(f"{name}:{GENDER_NAMES[genders[name]]}" for name in model_names),
                "male_probs": "|".join(f"{name}:{model_predictions[name][sample_id]['male_prob']:.6f}" for name in model_names),
                "min_confidence": min(confidences.values()),
                "max_confidence": max(confidences.values()),
                "confidence_gap": max(confidences.values()) - min(confidences.values()),
            }
        )
    rows.sort(key=lambda row: (-int(row["num_wrong_models"]), -float(row["min_confidence"]), str(row["dataset"]), str(row["sample_id"])))
    return rows


def write_predictions(
    output_path: Path,
    items: list[dict[str, Any]],
    model_predictions: dict[str, dict[str, dict[str, Any]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            sample_id = str(item["sample_id"])
            record = {
                "sample_id": sample_id,
                "dataset": item.get("dataset", ""),
                "image_path": item["image_path"],
                "label_gender": int(item["gender"]),
                "label_gender_name": GENDER_NAMES[int(item["gender"])],
                "label_age": float(item["age"]),
                "models": {name: predictions[sample_id] for name, predictions in model_predictions.items()},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_disagreements_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "dataset",
        "image_path",
        "label_gender",
        "label_gender_name",
        "label_age",
        "correct_models",
        "wrong_models",
        "num_correct_models",
        "num_wrong_models",
        "gender_predictions",
        "male_probs",
        "min_confidence",
        "max_confidence",
        "confidence_gap",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_focused_csvs(
    output_dir: Path,
    items: list[dict[str, Any]],
    model_predictions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, int]:
    focused_dir = output_dir / "focused"
    focused_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(model_predictions)
    counts: dict[str, int] = {}
    for left_idx, left in enumerate(model_names):
        for right in model_names[left_idx + 1 :]:
            rows = build_rows_for_models(items, model_predictions, [left, right])
            stem = f"{left}_vs_{right}"
            write_disagreements_csv(focused_dir / f"{stem}.csv", rows)
            counts[stem] = len(rows)
    aliases = {
        "public_vs_our_large": ("public_fairface_onnx", "our_large128_imdb_distill"),
        "teacher_vs_our_large": ("teacher_v2s_imdb", "our_large128_imdb_distill"),
        "mivolo_vs_our_large": ("mivolo_imdb_face", "our_large128_imdb_distill"),
        "mivolo_vs_public_fairface": ("mivolo_imdb_face", "public_fairface_onnx"),
    }
    available = set(model_names)
    for alias, pair in aliases.items():
        if pair[0] not in available or pair[1] not in available:
            continue
        rows = build_rows_for_models(items, model_predictions, [pair[0], pair[1]])
        write_disagreements_csv(focused_dir / f"{alias}.csv", rows)
        counts[alias] = len(rows)
    return counts


def build_rows_for_models(
    items: list[dict[str, Any]],
    model_predictions: dict[str, dict[str, dict[str, Any]]],
    model_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        sample_id = str(item["sample_id"])
        label = int(item["gender"])
        genders = {name: int(model_predictions[name][sample_id]["gender"]) for name in model_names}
        if len(set(genders.values())) <= 1:
            continue
        confidences = {name: float(model_predictions[name][sample_id]["gender_confidence"]) for name in model_names}
        correct = [name for name, gender in genders.items() if gender == label]
        wrong = [name for name, gender in genders.items() if gender != label]
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": str(item.get("dataset", "")),
                "image_path": str(item["image_path"]),
                "label_gender": label,
                "label_gender_name": GENDER_NAMES[label],
                "label_age": float(item["age"]),
                "correct_models": "|".join(correct),
                "wrong_models": "|".join(wrong),
                "num_correct_models": len(correct),
                "num_wrong_models": len(wrong),
                "gender_predictions": "|".join(f"{name}:{GENDER_NAMES[genders[name]]}" for name in model_names),
                "male_probs": "|".join(f"{name}:{model_predictions[name][sample_id]['male_prob']:.6f}" for name in model_names),
                "min_confidence": min(confidences.values()),
                "max_confidence": max(confidences.values()),
                "confidence_gap": max(confidences.values()) - min(confidences.values()),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["num_wrong_models"]),
            -float(row["min_confidence"]),
            str(row["dataset"]),
            str(row["sample_id"]),
        )
    )
    return rows


def render_contact_sheet(
    output_path: Path,
    rows: list[dict[str, Any]],
    max_images: int,
    thumb_size: int,
) -> None:
    selected = rows[:max_images]
    if not selected:
        return
    columns = 4
    cell_width = thumb_size
    cell_height = thumb_size + 104
    sheet_width = columns * cell_width
    sheet_height = math.ceil(len(selected) / columns) * cell_height
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(selected):
        x = (idx % columns) * cell_width
        y = (idx // columns) * cell_height
        try:
            image = Image.open(str(row["image_path"])).convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
        except Exception:
            image = Image.new("RGB", (thumb_size, thumb_size), "gray")
        sheet.paste(image, (x + (thumb_size - image.width) // 2, y))
        text_lines = [
            f"{row['sample_id']}",
            f"{row['dataset']} label={row['label_gender_name']} age={row['label_age']}",
            str(row["gender_predictions"])[:72],
            str(row["male_probs"])[:72],
        ]
        text_y = y + thumb_size + 4
        for line in text_lines:
            draw.text((x + 4, text_y), line, fill="black")
            text_y += 22
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def parse_model_specs(values: list[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 5:
            raise ValueError("--model must be name:kind:path:input_size:face_crop_margin")
        name, kind, path, input_size, margin = parts
        specs.append(
            ModelSpec(
                name=name,
                kind=kind,
                path=expand_path(path),
                input_size=int(input_size),
                face_crop_margin=float(margin),
            )
        )
    return specs


def parse_sample_limits(values: list[str] | None) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values or []:
        dataset, limit = value.split("=", 1)
        limits[dataset] = int(limit)
    return limits


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gender predictions across FastFace and public baseline models.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--model", action="append", required=True, help="name:kind:path:input_size:face_crop_margin")
    parser.add_argument("--sample-limit", action="append", help="dataset=count, applied after split filtering")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--public-batch-size", type=int, default=256)
    parser.add_argument("--mivolo-batch-size", type=int, default=128)
    parser.add_argument(
        "--mivolo-repo",
        type=Path,
        default=expand_path(os.environ["MIVOLO_REPO"]) if os.environ.get("MIVOLO_REPO") else None,
        help="Optional checkout of https://github.com/WildChlamydia/MiVOLO for mivolo_face models.",
    )
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-sheet-limit", type=int, default=80)
    args = parser.parse_args()

    specs = parse_model_specs(args.model)
    items = load_manifest_items(
        manifest_paths=args.manifest,
        split=args.split,
        sample_limits=parse_sample_limits(args.sample_limit),
        seed=args.seed,
    )
    if not items:
        raise ValueError("No manifest rows selected")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        if spec.kind == "fastface":
            model_predictions[spec.name] = predict_fastface(
                spec=spec,
                items=items,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        elif spec.kind == "fairface_onnx":
            model_predictions[spec.name] = predict_public_fairface_onnx(
                spec=spec,
                items=items,
                batch_size=args.public_batch_size,
            )
        elif spec.kind == "mivolo_face":
            model_predictions[spec.name] = predict_mivolo_face(
                spec=spec,
                items=items,
                batch_size=args.mivolo_batch_size,
                device=device,
                mivolo_repo=args.mivolo_repo,
            )
        else:
            raise ValueError(f"Unsupported model kind: {spec.kind}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(items, model_predictions)
    summary["manifests"] = [str(path) for path in args.manifest]
    summary["split"] = args.split
    summary["sample_limits"] = parse_sample_limits(args.sample_limit)
    summary["models_compared"] = [spec.__dict__ | {"path": str(spec.path)} for spec in specs]
    summary["selected_count"] = len(items)

    write_predictions(args.output_dir / "predictions.jsonl", items, model_predictions)
    disagreement_rows = build_rows(items, model_predictions)
    write_disagreements_csv(args.output_dir / "gender_disagreements.csv", disagreement_rows)
    focused_counts = write_pairwise_focused_csvs(args.output_dir, items, model_predictions)
    summary["focused_disagreements"] = focused_counts
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    render_contact_sheet(args.output_dir / "gender_disagreements_top.jpg", disagreement_rows, args.contact_sheet_limit, thumb_size=224)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"gender_disagreements={len(disagreement_rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
