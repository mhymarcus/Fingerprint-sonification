"""Compact U-Net for genuine ridge-mask prediction."""

try:
    import torch
    from torch import nn
except ImportError:  # The CV baseline remains usable without optional ML packages.
    torch = None
    nn = None


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for ML training/inference. "
            "Install the optional dependencies from requirements-ml.txt."
        )


if nn is not None:

    class _ConvBlock(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):
            return self.layers(inputs)


    class CompactUNet(nn.Module):
        """Small CPU-capable U-Net that predicts ridge logits, not an ROI mask."""

        def __init__(self, input_channels=1, output_channels=1, base_channels=24):
            super().__init__()
            self.encoder1 = _ConvBlock(input_channels, base_channels)
            self.encoder2 = _ConvBlock(base_channels, base_channels * 2)
            self.encoder3 = _ConvBlock(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _ConvBlock(base_channels * 4, base_channels * 8)
            self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
            self.decoder3 = _ConvBlock(base_channels * 8, base_channels * 4)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
            self.decoder2 = _ConvBlock(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
            self.decoder1 = _ConvBlock(base_channels * 2, base_channels)
            self.output = nn.Conv2d(base_channels, output_channels, 1)

        def forward(self, inputs):
            encoder1 = self.encoder1(inputs)
            encoder2 = self.encoder2(self.pool(encoder1))
            encoder3 = self.encoder3(self.pool(encoder2))
            bottleneck = self.bottleneck(self.pool(encoder3))
            decoder3 = self.decoder3(torch.cat((self.up3(bottleneck), encoder3), dim=1))
            decoder2 = self.decoder2(torch.cat((self.up2(decoder3), encoder2), dim=1))
            decoder1 = self.decoder1(torch.cat((self.up1(decoder2), encoder1), dim=1))
            return self.output(decoder1)

else:

    class CompactUNet:  # pragma: no cover - exercised only without optional torch.
        def __init__(self, *args, **kwargs):
            require_torch()
