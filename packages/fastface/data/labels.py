from __future__ import annotations

import math
from dataclasses import dataclass


AGE_BINS = 101
MAX_AGE = AGE_BINS - 1

FAIRFACE_AGE_RANGES: dict[str, tuple[int, int]] = {
    "0-2": (0, 2),
    "3-9": (3, 9),
    "10-19": (10, 19),
    "20-29": (20, 29),
    "30-39": (30, 39),
    "40-49": (40, 49),
    "50-59": (50, 59),
    "60-69": (60, 69),
    "more than 70": (70, 100),
    "70+": (70, 100),
}


@dataclass(frozen=True)
class AgeLabel:
    age: float
    age_min: int
    age_max: int
    label_type: str
    loss_weight: float


def normalize_gender(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in {"f", "female", "woman", "1"}:
        return 0
    if text in {"m", "male", "man", "0"}:
        return 1
    return None


def normalize_utk_gender(value: object) -> int | None:
    text = str(value).strip()
    if text == "1":
        return 0
    if text == "0":
        return 1
    return normalize_gender(value)


def clamp_age(age: float) -> float:
    if math.isnan(age):
        raise ValueError("age is NaN")
    return float(min(MAX_AGE, max(0.0, age)))


def exact_age(value: object, loss_weight: float = 1.0) -> AgeLabel | None:
    text = str(value).strip()
    if not text or text == "-1":
        return None
    age = clamp_age(float(text))
    rounded = int(round(age))
    return AgeLabel(age=age, age_min=rounded, age_max=rounded, label_type="exact", loss_weight=loss_weight)


def fairface_age(value: object, loss_weight: float = 0.35) -> AgeLabel | None:
    text = str(value).strip()
    age_range = FAIRFACE_AGE_RANGES.get(text)
    if age_range is None:
        return exact_age(text, loss_weight=1.0)
    age_min, age_max = age_range
    return AgeLabel(
        age=(age_min + age_max) / 2.0,
        age_min=age_min,
        age_max=age_max,
        label_type="range",
        loss_weight=loss_weight,
    )
