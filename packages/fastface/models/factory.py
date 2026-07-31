from __future__ import annotations

from typing import Any

from torch import nn

from fastface.models.age_gender_mobilenetv3 import AgeGenderMobileNetV3
from fastface.models.age_gender_torchvision import AgeGenderTorchvisionBackbone


def build_age_gender_model(model_cfg: dict[str, Any], pretrained: bool | None = None) -> nn.Module:
    name = str(model_cfg.get("name", "mobilenetv3_age_gender"))
    use_pretrained = bool(model_cfg.get("pretrained", True)) if pretrained is None else pretrained
    dropout = float(model_cfg.get("dropout", 0.2))

    if name == "mobilenetv3_age_gender":
        return AgeGenderMobileNetV3(
            variant=str(model_cfg.get("variant", "large")),
            pretrained=use_pretrained,
            dropout=dropout,
        )
    if name == "torchvision_age_gender":
        return AgeGenderTorchvisionBackbone(
            backbone=str(model_cfg["backbone"]),
            pretrained=use_pretrained,
            dropout=dropout,
        )
    raise ValueError(f"unsupported model name: {name}")
