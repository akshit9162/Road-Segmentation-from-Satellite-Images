# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------- Basic Conv Block -------
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


# ------- Attention Gate -------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi  # Scale skip-connection


# ------- Attention U-Net -------
class AttUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]

        self.conv1 = ConvBlock(3, filters[0])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = ConvBlock(filters[3], filters[4])

        # Attention gates
        self.att4 = AttentionGate(filters[3], filters[3], filters[2])
        self.att3 = AttentionGate(filters[2], filters[2], filters[1])
        self.att2 = AttentionGate(filters[1], filters[1], filters[0])

        # Up-sampling
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, 2)
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, 2)
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, 2)
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, 2)

        # Decoder conv blocks
        self.d4 = ConvBlock(filters[4], filters[3])
        self.d3 = ConvBlock(filters[3], filters[2])
        self.d2 = ConvBlock(filters[2], filters[1])
        self.d1 = ConvBlock(filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        # Decoder
        u4 = self.up4(c5)
        x4 = self.att4(g=u4, x=c4)
        d4 = self.d4(torch.cat([u4, x4], dim=1))

        u3 = self.up3(d4)
        x3 = self.att3(g=u3, x=c3)
        d3 = self.d3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(d3)
        x2 = self.att2(g=u2, x=c2)
        d2 = self.d2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        d1 = self.d1(torch.cat([u1, c1], dim=1))

        out = self.final(d1)
        return out
