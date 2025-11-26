import torch
import torch.nn as nn
import torch.nn.functional as F

from ta_mosc import MoE, SkipMoSCLayer


class ResidualBlock(nn.Module):
    """Residual conv block with identity mapping."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResidualEncoder(nn.Module):
    """Residual block followed by downsampling."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


class ResidualDecoder(nn.Module):
    """Upsample + concatenation + residual refinement."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_c * 2, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ReMoEUNet(nn.Module):
    """
    Residual U-Net with TA-MoSC enhanced skip connections.

    Combines residual feature extraction with MoE-refined skips.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        moe_embed_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        moe_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.moe_loss_weight = moe_loss_weight

        self.encoders = nn.ModuleList()
        in_c = in_channels
        for feat in features:
            self.encoders.append(ResidualEncoder(in_c, feat))
            in_c = feat

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)

        self.decoders = nn.ModuleList()
        dec_in = features[-1] * 2
        for feat in reversed(features):
            self.decoders.append(ResidualDecoder(dec_in, feat))
            dec_in = feat

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.moe = MoE(emb_size=moe_embed_dim, num_experts=num_experts, top_k=top_k)
        self.skip_adapters = nn.ModuleList(
            [SkipMoSCLayer(feat, moe_embed_dim) for feat in features]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips = []
        out = x
        for encoder in self.encoders:
            skip, out = encoder(out)
            skips.append(skip)

        out = self.bottleneck(out)

        aux_losses = []
        for decoder, skip_layer, skip_feat in zip(
            self.decoders, reversed(self.skip_adapters), reversed(skips)
        ):
            refined_skip, aux = skip_layer(skip_feat, self.moe)
            aux_losses.append(aux)
            out = decoder(out, refined_skip)

        logits = self.head(out)
        aux_total = torch.stack(aux_losses).sum() * self.moe_loss_weight
        return logits, aux_total


__all__ = ["ReMoEUNet"]

