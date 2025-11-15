# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RoadSegDataset
from models import AttUNet
import torch.optim as optim
from tqdm import tqdm


def dice_loss(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def train():
    device = "cuda" if torch.cuda.is_available() else "mps"
    print("Using device:", device)

    dataset = RoadSegDataset("/Users/akshitgupta/Desktop/datathon/data")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = AttUNet().to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            logits = logits.squeeze(1)
            masks = masks.squeeze(1)

            loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "att_unet_best.pth")
    print("Model saved as att_unet_best.pth")


if __name__ == "__main__":
    train()
