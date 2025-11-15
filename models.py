import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        c5 = self.bottleneck(p4)

        up4 = self.up4(c5)
        m4 = torch.cat([up4, c4], dim=1)
        c6 = self.dec4(m4)

        up3 = self.up3(c6)
        m3 = torch.cat([up3, c3], dim=1)
        c7 = self.dec3(m3)

        up2 = self.up2(c7)
        m2 = torch.cat([up2, c2], dim=1)
        c8 = self.dec2(m2)

        up1 = self.up1(c8)
        m1 = torch.cat([up1, c1], dim=1)
        c9 = self.dec1(m1)

        return self.final(c9)
