"""U-Net Radar Detector — RDM 기반 표적 탐지.

Architecture:
    Input: (B, 2, Nc, Nr) — RDM log-magnitude + phase
    -> Encoder: 5 stages, channels [32, 64, 128, 256, 512]
    -> Decoder: 4 stages with skip connections
    -> Conv 1x1 -> Sigmoid
    Output: (B, 1, Nc, Nr) — detection probability map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.train_utils import count_parameters


class ConvBlock(nn.Module):
    """Conv2d x 2 + BN + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDetector(nn.Module):
    """U-Net for radar target detection on Range-Doppler Maps.

    Parameters
    ----------
    in_channels : int
        Input channels (default: 2 for mag + phase).
    base_ch : int
        Base channel count (default: 32). Encoder doubles each stage.
    dropout : float
        Dropout in bottleneck (default: 0.3).
    """

    def __init__(self, in_channels=2, base_ch=32, dropout=0.3):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        # Encoder
        self.enc1 = ConvBlock(in_channels, ch[0])
        self.enc2 = ConvBlock(ch[0], ch[1])
        self.enc3 = ConvBlock(ch[1], ch[2])
        self.enc4 = ConvBlock(ch[2], ch[3])

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(ch[3], ch[4]),
            nn.Dropout2d(dropout),
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch[4], ch[3], 2, stride=2)
        self.dec4 = ConvBlock(ch[3] * 2, ch[3])

        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3 = ConvBlock(ch[2] * 2, ch[2])

        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = ConvBlock(ch[1] * 2, ch[1])

        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = ConvBlock(ch[0] * 2, ch[0])

        # Output
        self.out_conv = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, 2, H, W)

        Returns
        -------
        det : Tensor (B, 1, H, W), values in [0, 1]
        """
        _, _, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = torch.sigmoid(self.out_conv(d1))

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice for sparse target detection."""

    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, target):
        pred_c = pred.clamp(1e-6, 1 - 1e-6)
        bce = -target * torch.log(pred_c) - (1 - target) * torch.log(1 - pred_c)
        pt = target * pred_c + (1 - target) * (1 - pred_c)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        focal_loss = focal.mean()

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice

        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


if __name__ == '__main__':
    model = UNetDetector(in_channels=2, base_ch=32)
    x = torch.randn(2, 2, 128, 256)
    y = model(x)
    print(f"UNetDetector: input {x.shape} -> output {y.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
