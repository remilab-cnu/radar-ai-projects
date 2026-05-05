"""ResNet-18 Micro-Doppler HAR Model.

Architecture:
    Input: (B, 1, 128, 128) -- micro-Doppler spectrogram
    -> ResNet-18 (1ch input)
    -> Global Average Pool -> FC(n_classes)
    Output: (B, n_classes) -- class logits
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet BasicBlock: two 3x3 convs with skip connection."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 implementation for single-channel input.

    Parameters
    ----------
    in_channels : int
        Input channels (default: 1 for spectrogram).
    n_classes : int
        Number of output classes.
    """

    def __init__(self, in_channels=1, n_classes=6):
        super().__init__()
        self.in_ch = 64

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_ch, n_blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, 1, 128, 128) -> logits: (B, n_classes)"""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class ResNetHAR(nn.Module):
    """ResNet-18 wrapper for micro-Doppler HAR.

    Parameters
    ----------
    n_classes : int
        Number of activity classes.
    """

    def __init__(self, n_classes=6):
        super().__init__()
        self.model = ResNet18(in_channels=1, n_classes=n_classes)

    def forward(self, x):
        return self.model(x)


class TinyCNNHAR(nn.Module):
    """Compact CNN baseline for micro-Doppler spectrograms.

    This is intentionally much smaller than ResNet-18.  It gives the lecture a
    neural comparison point between handcrafted features and a deep residual
    model without making every result depend on an 11M-parameter network.
    """

    def __init__(self, n_classes=6, width=24, dropout=0.25):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, n_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_har_model(model_name="resnet18", n_classes=6):
    """Factory used by train/eval scripts."""
    if model_name == "resnet18":
        return ResNetHAR(n_classes=n_classes)
    if model_name == "tiny_cnn":
        return TinyCNNHAR(n_classes=n_classes)
    raise ValueError(f"unknown HAR model {model_name!r}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = ResNetHAR(n_classes=6)
    x = torch.randn(4, 1, 128, 128)
    y = model(x)
    print(f"ResNetHAR: input {x.shape} -> output {y.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
