# Road Segmentation from Satellite Images

Binary road segmentation of satellite imagery with a U-Net in PyTorch, built for a hackathon
(Nov 2025). Given an RGB satellite tile, the model predicts a per-pixel road mask.

## Pipeline

| Stage | Details | Code |
|---|---|---|
| Data | Image/mask pairs, random 224×224 patches for training | `dataset.py` (`RoadSegDataset`) |
| Model | U-Net: double-conv blocks (Conv–BatchNorm–ReLU ×2), 32 base filters, 1 output channel | `models.py` |
| Loss | BCE-with-logits + Dice loss, which handles the class imbalance (roads are a small fraction of pixels) | `train.py` |
| Optimiser | Adam, learning rate 1e-4, batch size 8 | `train.py` |
| Inference | Sliding-window over full-size images (256 px windows, 32 px overlap), stitched into one mask | `utils.py` (`sliding_window_inference`), `test.py` |
| Metrics | IoU and Dice (PyTorch and NumPy versions) | `utils.py` |
| Scoring | Foreground/background IoU, mean IoU and Dice over paired PNGs, after binarising and dilating masks | `hackathon_checker.py` |

## Run

```bash
pip install torch numpy pillow opencv-python tqdm
python train.py                                   # saves unet_best.pth
python test.py --model unet_best.pth --input_dir <images> --output_dir <masks>
python hackathon_checker.py                       # set RESULT_DIR / GT_DIR inside the script
```
