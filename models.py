# model.py
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        chs = [64, 128, 256, 512, 1024]

        self.conv00 = ConvBlock(3, chs[0])
        self.pool = nn.MaxPool2d(2)

        self.conv10 = ConvBlock(chs[0], chs[1])
        self.conv20 = ConvBlock(chs[1], chs[2])
        self.conv30 = ConvBlock(chs[2], chs[3])
        self.conv40 = ConvBlock(chs[3], chs[4])

        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, 2)
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, 2)
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, 2)
        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, 2)

        self.final = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        x31 = self.up4(x40)
        x21 = self.up3(x30 + x31)
        x11 = self.up2(x20 + x21)
        x01 = self.up1(x10 + x11)

        out = self.final(x01)
        return out  # returns ONLY final output
