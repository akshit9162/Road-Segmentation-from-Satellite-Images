# train.py
# MPS/CUDA-optimized training script with safer memory settings for macOS MPS.

import os
import glob
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import RoadSegDataset
from models import UNet
from utils import iou_pytorch, dice_pytorch

import albumentations as A

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = os.path.abspath("data")     # folder with both .jpg and _mask.png files
SAVE_DIR = "checkpoints"

# Lowered memory footprint:
PATCH_SIZE = 256        # keep patch size, but we resize to 256 in augmentations
BATCH_SIZE = 2          # <<-- reduce batch size to avoid OOM on MPS
EPOCHS = 25
LR = 1e-4
VAL_SPLIT = 0.1

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Device & performance hints
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# lower-mantissa matmul precision can help on MPS/CPU
try:
    torch.set_float32_matmul_precision('medium')
except Exception:
    pass

# -----------------------------
# Data preparation
# -----------------------------
all_files = sorted([
    os.path.splitext(f)[0].replace("_sat", "")  # remove "_sat" and ".jpg"
    for f in os.listdir(DATA_DIR)
    if f.endswith("_sat.jpg")
])

train_files, val_files = train_test_split(all_files, test_size=VAL_SPLIT, random_state=42)
print(f"Total: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

# -----------------------------
# Augmentations
# -----------------------------
# Resize added to reduce memory footprint. If you want higher res, increase but also increase memory.
augment_fn = A.Compose([
    A.Resize(256, 256),                           # <<-- resize to 256x256 to lower memory
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(0.04, 0.04), p=0.5)
])

# -----------------------------
# Datasets and loaders
# -----------------------------
train_ds = RoadSegDataset(
    data_dir=DATA_DIR,
    files=train_files,
    patch_size=PATCH_SIZE,
    is_train=True,
    augment_fn=augment_fn
)
val_ds = RoadSegDataset(
    data_dir=DATA_DIR,
    files=val_files,
    patch_size=PATCH_SIZE,
    is_train=False,
    augment_fn=None
)

# On macOS/MPS use num_workers=0 to avoid multiprocessing spawn issues
num_workers = 0
pin_memory = False

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# -----------------------------
# Model, loss, optimizer
# -----------------------------
# If you still run OOM, change base_filters to 16 (smaller model)
model = UNet(in_ch=3, out_ch=1, base_filters=32).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

# Disable AMP on MPS for stability / memory reasons
use_amp = False
scaler = None

# -----------------------------
# Training loop (wrapped in __main__ for safe multiprocessing)
# -----------------------------
if __name__ == "__main__":
    best_iou = 0.0

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
            for imgs, masks, _ in pbar:
                imgs = imgs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                if use_amp:
                    # (unused because use_amp=False)
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(imgs)
                        loss = criterion(outputs, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                # free up MPS caches (helpful on macOS)
                if device.type == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass

                train_losses.append(loss.detach().cpu())
                if len(train_losses) > 0:
                    pbar.set_postfix(loss=f"{float(torch.stack(train_losses).mean()):.4f}")

            # -----------------------------
            # Validation
            # -----------------------------
            model.eval()
            iou_acc = 0.0
            dice_acc = 0.0
            n_batches = 0

            with torch.no_grad():
                for imgs, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False):
                    imgs = imgs.to(device)
                    masks = masks.to(device)

                    outputs = model(imgs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    # compute metrics on-device, then accumulate
                    batch_iou = 0.0
                    batch_dice = 0.0
                    # average over batch
                    for p, t in zip(preds, masks):
                        batch_iou += iou_pytorch(p, t)
                        batch_dice += dice_pytorch(p, t)

                    batch_iou /= preds.shape[0]
                    batch_dice /= preds.shape[0]

                    iou_acc += batch_iou
                    dice_acc += batch_dice
                    n_batches += 1

            mean_iou = iou_acc / max(1, n_batches)
            mean_dice = dice_acc / max(1, n_batches)

            print(f"\nEpoch {epoch}: val IoU={mean_iou:.4f}  Dice={mean_dice:.4f}")

            # scheduler step uses the monitored metric
            scheduler.step(mean_iou)

            # save best
            if mean_iou > best_iou:
                best_iou = mean_iou
                ckpt = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_iou': mean_iou
                }
                torch.save(ckpt, os.path.join(SAVE_DIR, 'best.pth'))
                print(f"Saved best model (IoU={best_iou:.4f})")

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        torch.save({'model_state': model.state_dict()}, os.path.join(SAVE_DIR, 'interrupted.pth'))

    print("Training finished.")
