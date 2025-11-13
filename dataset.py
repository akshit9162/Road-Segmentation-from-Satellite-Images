import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import torch

class RoadSegDataset(Dataset):
    def __init__(self, data_dir, files=None, patch_size=None, is_train=True, augment_fn=None):
        self.images_dir = data_dir
        self.masks_dir = data_dir
        self.patch_size = patch_size
        self.is_train = is_train
        self.augment_fn = augment_fn

        # collect valid pairs (image: *_sat.jpg, mask: *_mask.png)
        all_imgs = sorted([f for f in os.listdir(data_dir) if f.endswith("_sat.jpg")])
        self.valid_basenames = []
        for f in all_imgs:
            base = f.replace("_sat.jpg", "")
            mask_path = os.path.join(data_dir, f"{base}_mask.png")
            if os.path.exists(mask_path):
                self.valid_basenames.append(base)

        if files is not None:
            # subset to provided split
            self.valid_basenames = [b for b in self.valid_basenames if b in files]

        print(f"Total valid pairs found: {len(self.valid_basenames)}")

    def __len__(self):
        return len(self.valid_basenames)

    def _load_pair(self, basename):
        img_path = os.path.join(self.images_dir, f"{basename}_sat.jpg")
        mask_path = os.path.join(self.masks_dir, f"{basename}_mask.png")

        # load and binarize mask
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 128).astype(np.float32)  # binarize
        return img, mask

    def __getitem__(self, idx):
        basename = self.valid_basenames[idx]
        img, mask = self._load_pair(basename)

        # apply augmentations
        if self.augment_fn:
            augmented = self.augment_fn(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0), basename

