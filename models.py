import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Basic Helpers
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


# ------------------------------
# Attention Gate
# ------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ------------------------------
# Attention U-Net
# ------------------------------
class AttnUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, base_filters=64):
        super().__init__()

        filters = [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]

        # Encoder
        self.conv1 = ConvBlock(input_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = ConvBlock(filters[3], filters[4])

        # Attention + Up
        self.up4 = UpConv(filters[4], filters[3])
        self.att4 = AttentionBlock(filters[3], filters[3], filters[2])
        self.up_conv4 = ConvBlock(filters[4], filters[3])

        self.up3 = UpConv(filters[3], filters[2])
        self.att3 = AttentionBlock(filters[2], filters[2], filters[1])
        self.up_conv3 = ConvBlock(filters[3], filters[2])

        self.up2 = UpConv(filters[2], filters[1])
        self.att2 = AttentionBlock(filters[1], filters[1], filters[0])
        self.up_conv2 = ConvBlock(filters[2], filters[1])

        self.up1 = UpConv(filters[1], filters[0])
        self.att1 = AttentionBlock(filters[0], filters[0], filters[0] // 2)
        self.up_conv1 = ConvBlock(filters[1], filters[0])

        # Final
        self.out_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))

        # Decoder with attentions
        d4 = self.up4(x5)
        x4 = self.att4(d4, x4)
        d4 = torch.cat([x4, d4], dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(d3, x3)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(d2, x2)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(d1, x1)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.up_conv1(d1)

        return self.out_conv(d1)
