from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from fastface.data.manifest_dataset import ManifestDataset
from fastface.models.factory import build_age_gender_model
from fastface.paths import expand_path


@dataclass
class DistState:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(os.path.expandvars(handle.read()))


def setup_dist() -> DistState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return DistState(distributed=distributed, rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup_dist(state: DistState) -> None:
    if state.distributed and dist.is_initialized():
        dist.destroy_process_group()


def seed_all(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1)


def reduce_sum(values: torch.Tensor, state: DistState) -> torch.Tensor:
    if state.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def make_loader(config: dict[str, Any], split: str, training: bool, state: DistState) -> tuple[DataLoader, DistributedSampler | None]:
    data_cfg = config["data"]
    sample_limits_cfg = data_cfg.get("sample_limits", {})
    sample_limits = sample_limits_cfg.get(split, {}) if isinstance(sample_limits_cfg, dict) else {}
    manifest_paths = data_cfg.get(f"{split}_manifests", data_cfg["manifests"])
    dataset = ManifestDataset(
        manifest_paths=[expand_path(p) for p in manifest_paths],
        split=split,
        input_size=int(data_cfg.get("input_size", 128)),
        training=training,
        age_sigma=float(data_cfg.get("age_sigma", 2.0)),
        face_crop_margin=float(data_cfg.get("face_crop_margin", 0.0)),
        sample_limits={str(key): int(value) for key, value in sample_limits.items()},
        seed=int(config.get("seed", 20260730)),
    )
    sampler = DistributedSampler(dataset, shuffle=training) if state.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(data_cfg.get("num_workers", 8)) > 0,
        drop_last=training,
    )
    return loader, sampler


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = config.get("model", {})
    return build_age_gender_model(model_cfg).to(device)


def load_initial_checkpoint(model: nn.Module, config: dict[str, Any]) -> None:
    train_cfg = config.get("train", {})
    checkpoint_path = train_cfg.get("initial_checkpoint")
    if not checkpoint_path:
        return
    checkpoint = torch.load(expand_path(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])


def load_teacher(config: dict[str, Any], device: torch.device) -> nn.Module | None:
    distill_cfg = config.get("distillation")
    if not distill_cfg or not bool(distill_cfg.get("enabled", False)):
        return None
    checkpoint_path = expand_path(distill_cfg["teacher_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher_config = checkpoint["config"]
    teacher = build_model(teacher_config, device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def kl_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)


def train_one_epoch(
    model: nn.Module,
    teacher: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    state: DistState,
    config: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_cfg = config.get("loss", {})
    gender_weight = float(loss_cfg.get("gender_weight", 2.0))
    age_weight = float(loss_cfg.get("age_weight", 1.0))
    distill_cfg = config.get("distillation", {})
    distill_temperature = float(distill_cfg.get("temperature", 2.0))
    distill_gender_weight = float(distill_cfg.get("gender_weight", 0.0))
    distill_age_weight = float(distill_cfg.get("age_weight", 0.0))
    amp_enabled = bool(config.get("train", {}).get("amp", True)) and state.device.type == "cuda"
    totals = torch.zeros(8, device=state.device)
    iterator = tqdm(loader, desc=f"epoch {epoch} train", disable=state.rank != 0)
    for batch in iterator:
        image = batch["image"].to(state.device, non_blocking=True)
        gender = batch["gender"].to(state.device, non_blocking=True)
        age_target = batch["age_target"].to(state.device, non_blocking=True)
        age_loss_weight = batch["age_loss_weight"].to(state.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=state.device.type, enabled=amp_enabled):
            output = model(image)
            gender_loss_each = F.cross_entropy(output["gender_logits"], gender, reduction="none")
            age_loss_each = soft_cross_entropy(output["age_logits"], age_target) * age_loss_weight
            loss = gender_weight * gender_loss_each.mean() + age_weight * age_loss_each.mean()
            distill_gender_loss = output["gender_logits"].new_tensor(0.0)
            distill_age_loss = output["age_logits"].new_tensor(0.0)
            if teacher is not None and (distill_gender_weight > 0.0 or distill_age_weight > 0.0):
                with torch.no_grad():
                    teacher_output = teacher(image)
                distill_gender_loss = kl_distillation_loss(
                    output["gender_logits"], teacher_output["gender_logits"], distill_temperature
                )
                distill_age_loss = kl_distillation_loss(
                    output["age_logits"], teacher_output["age_logits"], distill_temperature
                )
                loss = loss + distill_gender_weight * distill_gender_loss + distill_age_weight * distill_age_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predicted_gender = output["gender_logits"].argmax(dim=1)
        batch_size = gender.numel()
        totals[0] += loss.detach() * batch_size
        totals[1] += (predicted_gender == gender).sum()
        totals[2] += batch_size
        totals[3] += gender_loss_each.detach().sum()
        totals[4] += age_loss_each.detach().sum()
        totals[5] += batch_size
        totals[6] += distill_gender_loss.detach() * batch_size
        totals[7] += distill_age_loss.detach() * batch_size
    totals = reduce_sum(totals, state)
    count = max(float(totals[2].item()), 1.0)
    return {
        "train_loss": float(totals[0].item() / count),
        "train_gender_acc": float(totals[1].item() / count),
        "train_gender_loss": float(totals[3].item() / count),
        "train_age_loss": float(totals[4].item() / count),
        "train_distill_gender_loss": float(totals[6].item() / count),
        "train_distill_age_loss": float(totals[7].item() / count),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, state: DistState) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(10, device=state.device)
    for batch in tqdm(loader, desc="val", disable=state.rank != 0):
        image = batch["image"].to(state.device, non_blocking=True)
        gender = batch["gender"].to(state.device, non_blocking=True)
        age = batch["age"].to(state.device, non_blocking=True)
        output = model(image)
        predicted_gender = output["gender_logits"].argmax(dim=1)
        predicted_age = output["age"]
        abs_error = (predicted_age - age).abs()
        totals[0] += (predicted_gender == gender).sum()
        totals[1] += gender.numel()
        for cls in (0, 1):
            mask = gender == cls
            totals[2 + cls] += ((predicted_gender == gender) & mask).sum()
            totals[4 + cls] += mask.sum()
        totals[6] += abs_error.sum()
        totals[7] += age.numel()
        totals[8] += (abs_error <= 5.0).sum()
        totals[9] += F.cross_entropy(output["gender_logits"], gender, reduction="sum")
    totals = reduce_sum(totals, state)
    count = max(float(totals[1].item()), 1.0)
    female_count = max(float(totals[4].item()), 1.0)
    male_count = max(float(totals[5].item()), 1.0)
    female_acc = float(totals[2].item() / female_count)
    male_acc = float(totals[3].item() / male_count)
    age_count = max(float(totals[7].item()), 1.0)
    return {
        "val_gender_acc": float(totals[0].item() / count),
        "val_gender_balanced_acc": (female_acc + male_acc) / 2.0,
        "val_female_acc": female_acc,
        "val_male_acc": male_acc,
        "val_age_mae": float(totals[6].item() / age_count),
        "val_age_cs5": float(totals[8].item() / age_count),
        "val_gender_loss": float(totals[9].item() / count),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FastFace age/gender model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    state = setup_dist()
    try:
        seed_all(int(config.get("seed", 20260730)), state.rank)
        train_loader, train_sampler = make_loader(config, split="train", training=True, state=state)
        val_loader, _ = make_loader(config, split="val", training=False, state=state)
        model = build_model(config, state.device)
        load_initial_checkpoint(model, config)
        teacher = load_teacher(config, state.device)
        if state.distributed:
            model = DistributedDataParallel(model, device_ids=[state.local_rank] if state.device.type == "cuda" else None)

        train_cfg = config.get("train", {})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 0.001)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        epochs = int(train_cfg.get("epochs", 40))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and state.device.type == "cuda")
        output_dir = args.output_dir or Path(train_cfg.get("output_dir", "runs/age_gender_mobilenetv3"))
        output_dir.mkdir(parents=True, exist_ok=True)
        best_score = -math.inf

        if state.rank == 0:
            with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)

        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = train_one_epoch(model, teacher, train_loader, optimizer, scaler, state, config, epoch)
            val_metrics = evaluate(model, val_loader, state)
            scheduler.step()
            metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}
            score = metrics["val_gender_balanced_acc"] - 0.002 * metrics["val_age_mae"]
            if state.rank == 0:
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, config)
                if score > best_score:
                    best_score = score
                    save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, config)
                print(json.dumps(metrics, sort_keys=True), flush=True)
    finally:
        cleanup_dist(state)


if __name__ == "__main__":
    main()
