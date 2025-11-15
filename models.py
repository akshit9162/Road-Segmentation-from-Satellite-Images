# unetpp_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetPP(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        nb = [64, 128, 256, 512, 1024]

        self.c0_0 = ConvBlock(3, nb[0])
        self.c1_0 = ConvBlock(nb[0], nb[1])
        self.c2_0 = ConvBlock(nb[1], nb[2])
        self.c3_0 = ConvBlock(nb[2], nb[3])
        self.c4_0 = ConvBlock(nb[3], nb[4])

        self.c0_1 = ConvBlock(nb[0] + nb[1], nb[0])
        self.c1_1 = ConvBlock(nb[1] + nb[2], nb[1])
        self.c2_1 = ConvBlock(nb[2] + nb[3], nb[2])
        self.c3_1 = ConvBlock(nb[3] + nb[4], nb[3])

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.final = nn.Conv2d(nb[0], n_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.c0_0(x)
        x1_0 = self.c1_0(self.pool(x0_0))
        x2_0 = self.c2_0(self.pool(x1_0))
        x3_0 = self.c3_0(self.pool(x2_0))
        x4_0 = self.c4_0(self.pool(x3_0))

        x0_1 = self.c0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.c1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.c2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.c3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        return self.final(x0_1)
