import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two-layer conv block with BN + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    """
    UNet++ implementation with dense skip connections.

    Args:
        in_channels: number of input channels (default 3 for RGB images).
        out_channels: number of output channels (default 1 for binary mask).
        features: channel widths per encoder stage.
    """

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        features = features or [64, 128, 256, 512]
        depth = len(features)

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Setup upsampling layers per row
        row_channels = list(features)
        row_channels[-1] = features[-1] * 2
        self.up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            row_channels[i + 1],
                            features[i],
                            kernel_size=2,
                            stride=2,
                        )
                        for _ in range(depth - i - 1)
                    ]
                )
                for i in range(depth)
            ]
        )

        # Dense intermediate nodes
        self.nodes = nn.ModuleList()
        for i in range(depth - 1):
            row = nn.ModuleList()
            for col in range(1, depth - i):
                in_ch = (col + 1) * features[i]
                row.append(ConvBlock(in_ch, features[i]))
            self.nodes.append(row)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        depth = len(self.encoders)

        # Store intermediate tensors X(i, j)
        X = [[None] * (depth - i) for i in range(depth)]

        # Downsample path: compute X(i, 0)
        for i in range(depth):
            x = self.encoders[i](x)
            X[i][0] = x
            if i < depth - 1:
                x = self.pools[i](x)

        bottle = self.bottleneck(x)
        X[depth - 1][0] = bottle

        # Bottom-up dense skip computation
        for layer in reversed(range(depth - 1)):
            for col in range(1, depth - layer):
                up_x = self.up[layer][col - 1](X[layer + 1][col - 1])
                concat_list = [X[layer][k] for k in range(col)] + [up_x]
                node_block = self.nodes[layer][col - 1]
                X[layer][col] = node_block(torch.cat(concat_list, dim=1))

        out = self.final_conv(X[0][-1])
        return out


if __name__ == "__main__":
    model = UNetPP()
    sample = torch.randn(2, 3, 512, 512)
    pred = model(sample)
    print(pred.shape)

