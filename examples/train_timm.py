"""Folder-based timm training with cyclic TreemapMix and exact padding reduction."""

import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
import timm
from timm.data import create_transform
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
import yaml
from treemapmix import (
    TreemapMix,
    TreemapDataset,
    DistributedViewSampler,
    collate_sparse,
    AlphaScheduler,
    KScheduler,
)
from treemapmix.integrations.timm import create_loss, to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model")
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop after this many epochs, keeping the full schedule",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text()) if args.config else {}
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.model:
        cfg["model"] = args.model
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    device = (
        torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            init_method=os.getenv("TREEMAPMIX_INIT_METHOD", "env://"),
            rank=rank,
            world_size=world,
        )
    torch.set_num_threads(int(cfg.get("cpu_threads", 2)))
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    epochs = int(cfg.get("epochs", 2))
    size = int(cfg.get("image_size", 256))
    if args.smoke:
        size = 64
        classes = max(4, int(cfg.get("k", 4)))
        base = TensorDataset(
            torch.rand(2 * classes + 1, 3, size, size),
            torch.arange(2 * classes + 1) % classes,
        )
        validation = TensorDataset(
            torch.rand(classes + 1, 3, size, size), torch.arange(classes + 1) % classes
        )
        targets = base.tensors[1].tolist()
    else:
        if args.data is None:
            parser.error("--data is required unless --smoke is selected")
        base = ImageFolder(
            args.data / "train",
            transform=create_transform(
                input_size=(3, size, size),
                is_training=True,
                auto_augment=cfg.get("auto_augment", "rand-m6-inc1-mstd1.0-n3"),
                re_prob=cfg.get("reprob", 0.2),
            ),
        )
        validation = ImageFolder(
            args.data / "val",
            transform=create_transform(
                input_size=(3, size, size), is_training=False, crop_pct=0.95
            ),
        )
        if base.class_to_idx != validation.class_to_idx:
            raise ValueError("Train/val class mapping mismatch")
        classes = len(base.classes)
        targets = base.targets
    loss_name = cfg.get("loss", "pl")
    beta = cfg.get("beta", 0.5)
    mix = TreemapMix(
        k=int(cfg.get("k", 4)),
        alpha=float(cfg.get("alpha_min", 0.1)),
        require_distinct=loss_name != "sce"
        and not (loss_name == "mixed" and beta == 1),
        duplicate_policy=cfg.get("duplicate_policy", "error"),
        sampling_enabled=cfg.get("sampling_enabled", True),
        max_aspect=cfg.get("max_aspect", 10.0),
        min_side=cfg.get("min_side", 0),
        max_attempts=cfg.get("max_attempts", 200),
    )
    cycles = 3 if args.smoke else cfg.get("cycles")
    row_count = cycles or max(
        1,
        int(
            np.ceil(
                len(base)
                / (
                    len(set(targets))
                    if cfg.get("grouping", "balanced") == "balanced"
                    else mix.mix_k
                )
            )
        ),
    )
    alpha = AlphaScheduler(
        cfg.get("alpha_min", 0.1),
        cfg.get("alpha_max", 0.5),
        "cosine",
        step_mode="total",
        total_epochs=epochs,
        cycles_per_epoch=row_count,
    )
    k_schedule = KScheduler(
        k_min=cfg.get("k_min", mix.mix_k),
        k_max=cfg.get("k_max", mix.mix_k),
        schedule=cfg.get("k_schedule", "fixed"),
        total_epochs=epochs,
        warmup_epochs=cfg.get("k_warmup_epochs", 0),
        cooldown_epochs=cfg.get("k_cooldown_epochs", 0),
    )
    dataset = TreemapDataset(
        base,
        targets,
        mix,
        mode=cfg.get("grouping", "balanced"),
        cycles=cycles,
        seed=seed,
        alpha_schedule=alpha,
        k_schedule=k_schedule,
    )
    model = timm.create_model(
        cfg.get("model", "resnet18") if not args.smoke else "resnet18",
        pretrained=False,
        num_classes=classes,
        drop_path_rate=cfg.get("drop_path", 0.0),
    ).to(device)
    optimizer = create_optimizer_v2(
        model,
        opt=cfg.get("optimizer", "nadamw"),
        lr=cfg.get("lr", 0.0015),
        weight_decay=cfg.get("weight_decay", 0.06),
    )
    warmup = min(int(cfg.get("warmup_epochs", 10)), max(0, epochs - 1))
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max(1, epochs - warmup),
        warmup_prefix=True,
        lr_min=cfg.get("min_lr", 5e-7),
        warmup_t=warmup,
        warmup_lr_init=cfg.get("warmup_lr", 5e-7),
    )
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda"
        and torch.cuda.get_device_capability()[0] >= 8
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"] + 1
        rng = checkpoint.get("rank_rng", [checkpoint])[
            rank if "rank_rng" in checkpoint else 0
        ]
        torch.set_rng_state(rng["torch_rng"])
        random.setstate(rng["python_rng"])
        np.random.set_state(rng["numpy_rng"])
        if device.type == "cuda" and rng.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(rng["cuda_rng"], device)
    wrapped = (
        DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
        if world > 1
        else model
    )
    criterion = create_loss(loss_name, beta, distributed=world > 1)
    batch_size = 4 if args.smoke else int(cfg.get("batch_size", 1024)) // world
    if not args.smoke and int(cfg.get("batch_size", 1024)) % world:
        raise ValueError("Global batch size must be divisible by world size")
    if batch_size < 1:
        raise ValueError("Global batch size must be at least world size")
    args.output.mkdir(parents=True, exist_ok=True)
    end = (
        min(epochs, start + args.stop_after) if args.stop_after is not None else epochs
    )
    for epoch in range(start, end):
        dataset.set_epoch(epoch)
        mix.groups = mix.unmixed_groups = 0
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=DistributedViewSampler(dataset, rank, world),
            num_workers=0 if args.smoke else int(cfg.get("workers", 0)),
            collate_fn=collate_sparse,
            pin_memory=device.type == "cuda",
        )
        wrapped.train()
        total_loss = total_valid = 0.0
        for batch in loader:
            images, target = to_device(batch, device)
            active = target[1].sum(1) > 0
            empty_rank = not active.any()
            if not empty_rank:
                images = images[active]
                target = (target[0][active], target[1][active])
            else:
                images = images[:1]
                target = (target[0][:1], target[1][:1])
            batch_norm = [
                layer
                for layer in model.modules()
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
            ]
            if empty_rank:
                for layer in batch_norm:
                    layer.eval()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = wrapped(images)
                loss = criterion(logits, target)
            if empty_rank:
                for layer in batch_norm:
                    layer.train()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            per_sample, valid = criterion.per_sample(logits.detach(), target)
            total_loss += per_sample.sum().item()
            total_valid += valid.sum().item()
        scheduler.step(epoch + 1)
        if world > 1:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        model.eval()
        correct = count = 0
        val_indices = list(range(rank, len(validation), world))
        with torch.no_grad():
            for images, labels in DataLoader(
                torch.utils.data.Subset(validation, val_indices), batch_size=batch_size
            ):
                predictions = model(images.to(device)).argmax(1).cpu()
                correct += (predictions == labels).sum().item()
                count += len(labels)
        metrics = torch.tensor(
            [total_loss, total_valid, correct, count],
            device=device,
            dtype=torch.float64,
        )
        if world > 1:
            dist.all_reduce(metrics)
        rng = {
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
        }
        rank_rng = [None] * world
        if world > 1:
            dist.all_gather_object(rank_rng, rng)
        else:
            rank_rng[0] = rng
        if rank == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": float(metrics[0] / metrics[1].clamp_min(1)),
                        "accuracy": float(metrics[2] / metrics[3].clamp_min(1)),
                        "valid_views": int(metrics[1]),
                        "unmixed_group_fraction": dataset.unmixed_fraction,
                    }
                )
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                    "rank_rng": rank_rng,
                    "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None,
                },
                args.output / "last.pt",
            )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
