import os
import sys
from typing import Tuple, List
import cv2
import numpy as np

#!/usr/bin/env python3
"""
hackathon_checker.py

Compare paired PNG segmentations in two folders (result vs ground truth).

- Edit RESULT_DIR and GT_DIR with hardcoded absolute or relative paths.
- Both folders must contain PNG files with identical names to be compared.
- Each image is:
    * read in grayscale
    * binarized with threshold 127 (>=128 -> white / 255)
    dice_scores = []
    * white foreground dilated with a circular 5x5 kernel
- For each pair the script computes:
    * IoU for foreground (white)
    * IoU for background (black)
    * mean IoU = (IoU_foreground + IoU_background) / 2
    * Dice score for foreground
- Prints per-image metrics and the overall average of mean IoU across the dataset.
"""


# === USER CONFIG: set hardcoded folders here ===
RESULT_DIR = "/home/aritra/Downloads/hackathon_dataset/gt_300"      # <-- change to your results folder
GT_DIR     = "/home/aritra/Downloads/hackathon_dataset/gt_300"  # <-- change to your ground-truth folder
# ================================================

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
THRESH = 127  # binarization threshold

def load_png_pairs(result_dir: str, gt_dir: str) -> List[Tuple[str, str, str]]:
    """Return list of (filename, result_path, gt_path) for matching PNG basenames."""
    res_files = {f for f in os.listdir(result_dir) if f.lower().endswith(".png")}
    gt_files = {f for f in os.listdir(gt_dir) if f.lower().endswith(".png")}
    common = sorted(res_files & gt_files)
    pairs = [(fname, os.path.join(result_dir, fname), os.path.join(gt_dir, fname)) for fname in common]
    return pairs

def read_binarize_dilate(path: str) -> np.ndarray:
    """Read image as grayscale, binarize at THRESH, dilate foreground, return boolean mask (True=foreground)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # threshold: pixels > THRESH become 255; others 0
    _, th = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY)
    # dilate foreground
    dil = cv2.dilate(th, KERNEL)
    mask = dil.astype(np.uint8) == 255
    return mask

def iou_and_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float, float]:
    
    """
    Compute IoU_foreground, IoU_background, Dice_foreground between two boolean masks.
    Returns (iou_fg, iou_bg, dice_fg).
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError("Masks must have the same shape for metric computation")
    a = mask_a.astype(np.bool_)
    b = mask_b.astype(np.bool_)
    # Foreground intersection / union
    inter_fg = np.logical_and(a, b).sum()
    union_fg = np.logical_or(a, b).sum()
    if union_fg == 0:
        iou_fg = 1.0  # both empty -> perfect overlap
    else:
        iou_fg = inter_fg / union_fg
    # Background: invert masks
    a_bg = np.logical_not(a)
    b_bg = np.logical_not(b)
    inter_bg = np.logical_and(a_bg, b_bg).sum()
    union_bg = np.logical_or(a_bg, b_bg).sum()
    if union_bg == 0:
        iou_bg = 1.0
    else:
        iou_bg = inter_bg / union_bg
    # Dice for foreground: 2*|A∩B| / (|A|+|B|)
    sum_sizes = a.sum() + b.sum()
    if sum_sizes == 0:
        dice_fg = 1.0
    else:
        dice_fg = 2 * inter_fg / sum_sizes
    return iou_fg, iou_bg, dice_fg

def main():
    pairs = load_png_pairs(RESULT_DIR, GT_DIR)
    if len(pairs) == 0:
        print("No matching PNG files found between the two folders.")
        sys.exit(1)

    mean_ious = []
    dice_scores = []
    print(f"Comparing {len(pairs)} images...\n")
    print("{:<40s} {:>8s} {:>8s} {:>10s}".format("filename", "IoU_fg", "IoU_bg", "Dice_fg"))
    print("-" * 70)
    for fname, res_path, gt_path in pairs:
        try:
            res_mask = read_binarize_dilate(res_path)
            gt_mask  = read_binarize_dilate(gt_path)
        except Exception as e:
            print(f"{fname}: ERROR reading or processing: {e}")
            continue
        if res_mask.shape != gt_mask.shape:
            print(f"{fname}: SKIP (size mismatch: {res_mask.shape} vs {gt_mask.shape})")
            continue
        iou_fg, iou_bg, dice_fg = iou_and_dice(res_mask, gt_mask)
        mean_iou = (iou_fg + iou_bg) / 2.0
        mean_ious.append(mean_iou)
        dice_scores.append(dice_fg)
        print("{:<40s} {:8.4f} {:8.4f} {:10.4f}".format(fname, iou_fg, iou_bg, dice_fg))

    if len(mean_ious) == 0:
        print("\nNo valid image pairs were processed.")
        sys.exit(1)

    overall_mean_iou = float(np.mean(mean_ious))
    print("\nOverall average of mean IoU across dataset: {:.6f}".format(overall_mean_iou))
    # Also print average Dice (foreground) across the dataset
    overall_mean_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    print("Overall average Dice (foreground) across dataset: {:.6f}".format(overall_mean_dice))

if __name__ == "__main__":
    main()