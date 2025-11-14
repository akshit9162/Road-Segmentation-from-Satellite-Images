#!/usr/bin/env python3
"""
train.py — full training pipeline for Attention U-Net (AttnUNet).
Supports CUDA (with AMP) and MPS/CPU fallback. Uses BCEWithLogits + Dice loss,
IoU/Dice metrics, checkpointing, optional torch.compile speedup, and safe dataloading.
"""

import os
import argparse
import random
import math
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as A

# Import project modules (expects these files to exist)
from dataset import RoadSegDataset
from models import AttnUNet

# -----------------------
# Utilities: metrics/loss
# -----------------------
def dice_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean().item()

def iou_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    inter = (preds * targets).sum(dim=1)
    union = (preds + targets - preds * targets).sum(dim=1)
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def dice_loss_logits(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def bce_dice_loss(logits, targets, bce_weight=0.5):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_logits(logits, targets)
    return bce * bce_weight + dloss * (1.0 - bce_weight)

# -----------------------
# Argument parser
# -----------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="data folder containing *_sat.jpg and *_mask.png")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base_filters", type=int, default=32, help="model width (reduce if OOM)")
    p.add_argument("--patch_size", type=int, default=None, help="patch_size for dataset (None => full/resized image)")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# -----------------------
# Setup seeds & device
# -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------
# Build augmentations
# -----------------------
def build_augs(resize_to=None):
    aug_list = []
    if resize_to is not None:
        aug_list.append(A.Resize(resize_to, resize_to))
    aug_list += [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.Affine(scale=(0.95, 1.05), rotate=(-15,15), translate_percent=(0.02,0.02), p=0.5)
    ]
    return A.Compose(aug_list)

# -----------------------
# Training / Validation
# -----------------------
def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, masks, _ in pbar:
        if device.type == "cuda":
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.detach().cpu().item())
        n += 1
        pbar.set_postfix(loss=f"{running_loss / n:.4f}")
        # free MPS caches if using mps
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return running_loss / max(1, n)

@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    losses = 0.0
    dices = []
    ious = []
    n = 0
    pbar = tqdm(loader, desc="Val", leave=False)
    for imgs, masks, _ in pbar:
        if device.type == "cuda":
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(imgs)
        else:
            logits = model(imgs)

        loss = bce_dice_loss(logits, masks)
        losses += float(loss.detach().cpu().item())
        dices.append(dice_from_logits(logits, masks))
        ious.append(iou_from_logits(logits, masks))
        n += 1
    mean_loss = losses / max(1, n)
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_loss, mean_dice, mean_iou

# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    set_seed(args.seed)
    data_dir = os.path.abspath(args.data)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = pick_device()
    print("Device:", device)
    # MPS: avoid num_workers > 0, pin_memory not supported
    if device.type == "mps":
        num_workers = 0
        pin_memory = False
    else:
        num_workers = args.num_workers
        pin_memory = True if device.type == "cuda" else False

    # Build file list from *_sat.jpg (strip _sat)
    all_basenames = sorted([
        os.path.splitext(f)[0].replace("_sat", "")
        for f in os.listdir(data_dir)
        if f.endswith("_sat.jpg")
    ])
    total = len(all_basenames)
    if total == 0:
        raise RuntimeError(f"No images found in {data_dir} matching '*_sat.jpg'")

    # split
    cutoff = int(total * (1.0 - args.val_split))
    train_files = all_basenames[:cutoff]
    val_files = all_basenames[cutoff:]
    print(f"Total: {total} | Train: {len(train_files)} | Val: {len(val_files)}")

    # Augmentations: resize via patch_size if provided, otherwise leave dataset to handle
    train_aug = build_augs(resize_to=args.patch_size)  # None or value
    val_aug = None

    # Datasets
    train_ds = RoadSegDataset(data_dir, files=train_files, patch_size=args.patch_size, is_train=True, augment_fn=train_aug)
    val_ds = RoadSegDataset(data_dir, files=val_files, patch_size=args.patch_size, is_train=False, augment_fn=val_aug)

    print(f"Total valid pairs found (train): {len(train_ds)}")
    print(f"Total valid pairs found (val):   {len(val_ds)}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0))

    # Model
    print("Building model...")
    model = AttnUNet(input_channels=3, num_classes=1, base_filters=args.base_filters).to(device)

    # For CUDA: channels_last + compile + AMP
    use_amp = False
    scaler = None
    if device.type == "cuda":
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
        model = model.to(memory_format=torch.channels_last)
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Training loop
    best_metric = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_loss, val_dice, val_iou = validate(model, val_loader, device, use_amp)

        print(f"Epoch {epoch}  TrainLoss: {train_loss:.4f}  ValLoss: {val_loss:.4f}  ValDice: {val_dice:.4f}  ValIoU: {val_iou:.4f}")

        # scheduler: we monitor val_iou (higher is better)
        scheduler.step(val_iou)

        # checkpoint best by ValDice then ValIoU
        metric = val_dice
        if metric > best_metric:
            best_metric = metric
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou
            }
            torch.save(ckpt, os.path.join(save_dir, "best_attnunet.pth"))
            print(f"Saved best model (ValDice={val_dice:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTraining complete — best ValDice: {best_metric:.4f}. Time: {elapsed/60:.2f} min")

if __name__ == "__main__":
    main()
