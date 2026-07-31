from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    ResNet18_Weights,
    Swin_T_Weights,
)
from torchvision.models import convnext_tiny, efficientnet_b0, efficientnet_v2_s, resnet18, swin_t


class AgeGenderTorchvisionBackbone(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True, age_bins: int = 101, dropout: float = 0.2) -> None:
        super().__init__()
        self.post_pool = nn.Identity()
        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            self.features = base.features
            self.avgpool = base.avgpool
        elif backbone == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            base = efficientnet_v2_s(weights=weights)
            in_features = base.classifier[1].in_features
            self.features = base.features
            self.avgpool = base.avgpool
        elif backbone == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = convnext_tiny(weights=weights)
            in_features = base.classifier[2].in_features
            self.features = base.features
            self.avgpool = base.avgpool
            self.post_pool = base.classifier[0]
        elif backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            in_features = base.fc.in_features
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
            )
            self.avgpool = base.avgpool
        elif backbone == "swin_t":
            weights = Swin_T_Weights.DEFAULT if pretrained else None
            base = swin_t(weights=weights)
            in_features = base.head.in_features
            self.features = nn.Sequential(base.features, base.norm, base.permute)
            self.avgpool = base.avgpool
        else:
            raise ValueError(f"unsupported torchvision backbone: {backbone}")

        self.neck = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
        )
        self.gender_head = nn.Linear(512, 2)
        self.age_head = nn.Linear(512, age_bins)
        self.register_buffer("age_values", torch.arange(age_bins, dtype=torch.float32), persistent=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(image)
        x = self.avgpool(x)
        x = self.post_pool(x)
        x = torch.flatten(x, 1)
        x = self.neck(x)
        gender_logits = self.gender_head(x)
        age_logits = self.age_head(x)
        age_probs = torch.softmax(age_logits, dim=1)
        age = (age_probs * self.age_values.to(age_probs.device)).sum(dim=1)
        return {
            "gender_logits": gender_logits,
            "age_logits": age_logits,
            "age": age,
        }
