# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RoadSegDataset
from models import UNetPP
from tqdm import tqdm
import os

DATA_DIR = "/Users/akshitgupta/Desktop/datathon/data"
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4

# -------------------------------------------------------------
# Loss Functions
# -------------------------------------------------------------
bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1,2))
    union = probs.sum(dim=(1,2)) + targets.sum(dim=(1,2))
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

def compute_iou(pred, target, eps=1e-7):
    inter = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return (inter + eps) / (union + eps)

# -------------------------------------------------------------
# Train Function
# -------------------------------------------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    dataset = RoadSegDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = UNetPP(n_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_iou = 0
        total_dice = 0

        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            masks = masks,float()

            logits = model(imgs).squeeze(1)

            loss = bce_loss(logits, masks) + dice_loss(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                dice = 1 - dice_loss(logits, masks)
                iou = compute_iou(preds.bool(), masks.bool())
                total_dice += dice.item()
                total_iou += iou.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}  Dice={total_dice/len(loader):.4f}  IoU={total_iou/len(loader):.4f}")

    torch.save(model.state_dict(), "unetpp_best.pth")
    print("Model saved: unetpp_best.pth")


if __name__ == "__main__":
    train()
