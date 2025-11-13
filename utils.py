# utils.py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# -----------------------------
# PYTORCH METRICS
# -----------------------------
def iou_pytorch(preds, targets, smooth=1e-6):
    """IoU metric for torch tensors"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def dice_pytorch(preds, targets, smooth=1e-6):
    """Dice metric for torch tensors"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

# -----------------------------
# NUMPY METRICS (for val.py)
# -----------------------------
def iou_numpy(pred, gt, smooth=1e-6):
    """IoU for numpy arrays"""
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (intersection + smooth) / (union + smooth)

def dice_numpy(pred, gt, smooth=1e-6):
    """Dice for numpy arrays"""
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

# -----------------------------
# SLIDING WINDOW INFERENCE
# -----------------------------
def sliding_window_inference(image, model, device, patch_size=256, overlap=32, batch_size=4):
    """
    Inference helper for large 1024x1024 images by tiling into patches.
    Returns a binary mask (0/1).
    """
    model.eval()
    h, w, _ = image.shape
    stride = patch_size - overlap
    out_mask = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    to_tensor = transforms.ToTensor()

    patches = []
    coords = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            patches.append(to_tensor(patch))
            coords.append((y, x))

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.stack(patches[i:i+batch_size]).to(device)
            preds = torch.sigmoid(model(batch)).cpu().numpy()
            for j, (y, x) in enumerate(coords[i:i+batch_size]):
                out_mask[y:y+patch_size, x:x+patch_size] += preds[j, 0]
                count_map[y:y+patch_size, x:x+patch_size] += 1

    out_mask = out_mask / np.maximum(count_map, 1e-6)
    out_mask = (out_mask > 0.5).astype(np.uint8)
    return out_mask
