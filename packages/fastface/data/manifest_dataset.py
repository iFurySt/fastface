from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from fastface.data.labels import AGE_BINS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if min(x0, y0, x1, y1) < 0 or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_face(image: Image.Image, item: dict[str, Any], margin: float) -> Image.Image:
    if margin <= 0.0:
        return image
    bbox = parse_bbox(item.get("bbox_face"))
    if bbox is None:
        return image
    x0, y0, x1, y1 = bbox
    width, height = image.size
    box_width = x1 - x0
    box_height = y1 - y0
    pad_x = box_width * margin
    pad_y = box_height * margin
    left = int(max(0.0, x0 - pad_x))
    upper = int(max(0.0, y0 - pad_y))
    right = int(min(float(width), x1 + pad_x))
    lower = int(min(float(height), y1 + pad_y))
    if right - left < 2 or lower - upper < 2:
        return image
    return image.crop((left, upper, right, lower))


def make_transforms(input_size: int, training: bool) -> transforms.Compose:
    if training:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def stable_dataset_seed(seed: int, dataset: str) -> int:
    value = seed
    for char in dataset:
        value = (value * 131 + ord(char)) % (2**32)
    return value


def age_distribution(age_min: int, age_max: int, age: float, label_type: str, sigma: float) -> torch.Tensor:
    values = torch.arange(AGE_BINS, dtype=torch.float32)
    if label_type == "range" and age_max > age_min:
        mask = (values >= age_min) & (values <= age_max)
        target = mask.float()
    else:
        target = torch.exp(-0.5 * ((values - float(age)) / sigma) ** 2)
    target = target / target.sum().clamp_min(1e-8)
    return target


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_paths: list[Path],
        split: str,
        input_size: int,
        training: bool,
        age_sigma: float = 2.0,
        face_crop_margin: float = 0.0,
        sample_limits: dict[str, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        for manifest_path in manifest_paths:
            with manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("split") == split:
                        self.items.append(item)
        if sample_limits:
            self.items = self.apply_sample_limits(self.items, sample_limits, seed)
        if not self.items:
            joined = ", ".join(str(p) for p in manifest_paths)
            raise ValueError(f"no rows for split={split!r} in {joined}")
        self.transform = make_transforms(input_size=input_size, training=training)
        self.age_sigma = age_sigma
        self.face_crop_margin = face_crop_margin

    @staticmethod
    def apply_sample_limits(
        items: list[dict[str, Any]],
        sample_limits: dict[str, int],
        seed: int,
    ) -> list[dict[str, Any]]:
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_dataset.setdefault(str(item.get("dataset", "")), []).append(item)
        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        for dataset, source_items in by_dataset.items():
            limit = sample_limits.get(dataset)
            if limit is not None and limit > 0 and len(source_items) > limit:
                rng = random.Random(stable_dataset_seed(seed, dataset))
                source_items = rng.sample(source_items, limit)
            for item in source_items:
                selected_ids.add(id(item))
        for item in items:
            if id(item) in selected_ids:
                selected.append(item)
        return selected

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = Image.open(item["image_path"]).convert("RGB")
        image = crop_face(image, item, margin=self.face_crop_margin)
        image_tensor = self.transform(image)
        age_target = age_distribution(
            age_min=int(item["age_min"]),
            age_max=int(item["age_max"]),
            age=float(item["age"]),
            label_type=str(item.get("age_label_type", "exact")),
            sigma=self.age_sigma,
        )
        return {
            "image": image_tensor,
            "gender": torch.tensor(int(item["gender"]), dtype=torch.long),
            "age_target": age_target,
            "age": torch.tensor(float(item["age"]), dtype=torch.float32),
            "age_loss_weight": torch.tensor(float(item.get("age_loss_weight", 1.0)), dtype=torch.float32),
            "dataset": item.get("dataset", ""),
        }
