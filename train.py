# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RoadSegDataset
from models import UNet
from tqdm import tqdm

# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss(pred, target, eps=1e-7):
    """Dice loss for binary segmentation"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

# -----------------------------
# Metrics
# -----------------------------
def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean().item()

def dice_score(pred, target, threshold=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return (2 * intersection / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps)).mean().item()

# -----------------------------
# Training function
# -----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # Dataset
    dataset = RoadSegDataset("/Users/akshitgupta/Desktop/datathon/data",
                             files=[f.replace("_sat.jpg", "") for f in os.listdir("/Users/akshitgupta/Desktop/datathon/data") if f.endswith("_sat.jpg")],
                             patch_size=224, is_train=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = UNet(in_ch=3, out_ch=1, base_filters=32).to(device)

    # Loss + Optimizer
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)

            loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Optional: compute metrics on one batch for monitoring
        with torch.no_grad():
            sample_logits, sample_masks = logits, masks
            iou = iou_score(sample_logits, sample_masks)
            dice = dice_score(sample_logits, sample_masks)
            print(f"Sample Metrics - IoU: {iou:.4f}, Dice: {dice:.4f}")

    torch.save(model.state_dict(), "unet_best.pth")
    print("Model saved as unet_best.pth")


if __name__ == "__main__":
    import os
    train()
