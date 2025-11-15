# dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

class RoadSegDataset(Dataset):
    def __init__(self, data_dir, img_size=512):
        self.data_dir = data_dir
        self.img_size = img_size

        all_imgs = sorted([f for f in os.listdir(data_dir) if f.endswith("_sat.jpg")])
        self.basenames = []

        for imgfile in all_imgs:
            base = imgfile.replace("_sat.jpg", "")
            maskfile = f"{base}_mask.png"
            if os.path.exists(os.path.join(data_dir, maskfile)):
                self.basenames.append(base)

        print("Total valid pairs:", len(self.basenames))

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        base = self.basenames[idx]

        img_path = os.path.join(self.data_dir, base + "_sat.jpg")
        mask_path = os.path.join(self.data_dir, base + "_mask.png")

        # RGB image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(0, 1, 2).permute(2, 0, 1)  # (3,H,W)

        # Grayscale mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype(np.float32)  # convert 0/255 → 0/1
        mask = torch.tensor(mask).unsqueeze(0)  # (1,H,W)

        return img, mask
