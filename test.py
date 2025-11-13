# test.py
import os, glob
import torch
from models import UNet
from utils import sliding_window_inference
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="path to checkpoint (best.pth)")
parser.add_argument("--input_dir", required=True, help="folder with .jpg test images")
parser.add_argument("--output_dir", required=True, help="folder to save predicted .png masks")
parser.add_argument("--device", default="cuda", help="cuda or cpu")
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--overlap", type=int, default=32)
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
ckp = torch.load(args.model, map_location=device)
model = UNet(in_ch=3, out_ch=1, base_filters=32).to(device)
model.load_state_dict(ckp['model_state'])
model.eval()

os.makedirs(args.output_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")))

for f in files:
    basename = os.path.splitext(os.path.basename(f))[0]
    img = Image.open(f).convert("RGB")
    img_np = np.array(img)
    mask = sliding_window_inference(img_np, model, device,
                                    patch_size=args.patch_size, overlap=args.overlap, batch_size=4)
    # mask is 0/1 uint8 -> convert to 0/255 PNG
    mask_255 = (mask * 255).astype('uint8')
    out_path = os.path.join(args.output_dir, basename + ".png")
    Image.fromarray(mask_255).save(out_path)
    print("Saved", out_path)
