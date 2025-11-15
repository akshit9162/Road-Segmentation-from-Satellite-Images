import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class RoadSegDataset(Dataset):
    def __init__(self, root):
        self.root = root

        # Find all satellite-mask pairs automatically
        files = sorted(os.listdir(root))
        sats = [f for f in files if f.endswith("_sat.jpg")]
        masks = [f.replace("_sat.jpg", "_mask.png") for f in sats]

        # Validate existence
        self.pairs = []
        for s, m in zip(sats, masks):
            if os.path.exists(os.path.join(root, m)):
                self.pairs.append((s, m))

        print("Total valid pairs:", len(self.pairs))

        # Transforms
        self.img_tf = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])

        self.mask_tf = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sat_name, mask_name = self.pairs[idx]

        img = Image.open(os.path.join(self.root, sat_name)).convert("RGB")
        mask = Image.open(os.path.join(self.root, mask_name)).convert("L")

        img = self.img_tf(img)
        mask = self.mask_tf(mask)

        return img, mask
