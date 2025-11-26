import torch
import torch.nn as nn

from ta_mosc import MoE, SkipMoSCLayer


class ConvBlock(nn.Module):
    """Two stacked conv layers with BN and ReLU."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Conv block followed by 2x2 max pooling."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class DecoderBlock(nn.Module):
    """Upsample with transpose conv then apply ConvBlock."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UTANet(nn.Module):
    """
    U-Net 风格的 UTANet，实现 TA-MoSC 融合的跳连。
    返回预测和 MoE 辅助损失，训练时应当将二者相加。
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
        self.features = features
        self.moe_loss_weight = moe_loss_weight

        self.encoders = nn.ModuleList()
        in_c = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(in_c, f))
            in_c = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.decoders = nn.ModuleList()
        dec_in_c = features[-1] * 2
        for f in reversed(features):
            self.decoders.append(DecoderBlock(dec_in_c, f))
            dec_in_c = f

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.moe = MoE(
            emb_size=moe_embed_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.skip_adapters = nn.ModuleList(
            [SkipMoSCLayer(f, moe_embed_dim) for f in features]
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
            refined_skip, aux_loss = skip_layer(skip_feat, self.moe)
            aux_losses.append(aux_loss)
            out = decoder(out, refined_skip)

        logits = self.head(out)
        aux_total = torch.stack(aux_losses).sum() * self.moe_loss_weight
        return logits, aux_total

