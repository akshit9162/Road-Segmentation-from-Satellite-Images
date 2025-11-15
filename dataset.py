# dataset.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class RoadSegDataset(Dataset):
    """
    Optimized, memory-safe dataset:
    - Correct magic methods (__init__, __len__, __getitem__)
    - Fast PIL resize before numpy conversion
    - Optional albumentations augment_fn
    """

    def __init__(self, data_dir, files, patch_size=224, is_train=True, augment_fn=None):
        self.data_dir = data_dir
        self.files = files
        self.patch_size = patch_size
        self.is_train = is_train
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.files)

    def _load_pair(self, basename):
        img_path = os.path.join(self.data_dir, f"{basename}_sat.jpg")
        mask_path = os.path.join(self.data_dir, f"{basename}_mask.png")

        # ---- LOAD IMAGE ----
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if self.patch_size:
                im = im.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            img = np.asarray(im, dtype=np.uint8)

        # ---- LOAD MASK ----
        with Image.open(mask_path) as m:
            m = m.convert("L")
            if self.patch_size:
                m = m.resize((self.patch_size, self.patch_size), Image.NEAREST)
            mask = np.asarray(m, dtype=np.uint8)

        return img, mask

    def __getitem__(self, idx):
        basename = self.files[idx]
        img, mask = self._load_pair(basename)

        # Albumentations augmentations (optional)
        if self.augment_fn is not None and self.is_train:
            augmented = self.augment_fn(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Normalize
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # To CHW tensors
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()

        return img, mask, basename
