# dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class RoadSegDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        all_imgs = sorted([f for f in os.listdir(data_dir) if f.endswith("_sat.jpg")])

        self.basenames = []
        for imgfile in all_imgs:
            base = imgfile.replace("_sat.jpg", "")
            if os.path.exists(os.path.join(data_dir, base + "_mask.png")):
                self.basenames.append(base)

        print("Total valid pairs:", len(self.basenames))

        self.img_tf = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])

        # Mask transform → grayscale only
        self.mask_tf = T.Compose([
            T.Resize((512, 512), interpolation=Image.NEAREST),
            T.Grayscale(num_output_channels=1),
        ])

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        base = self.basenames[idx]

        img = Image.open(os.path.join(self.data_dir, base + "_sat.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.data_dir, base + "_mask.png")).convert("RGB")

        img = self.img_tf(img)

        mask = self.mask_tf(mask)              # → shape [1,512,512]
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        # If mask is (1,H,W) convert to (H,W)
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        # convert to binary 0/1
        mask = (mask > 127).long()

        return img, mask
