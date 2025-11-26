"""
Task-Adaptive Mixture of Skip Connections (TA-MoSC) utilities.

This module implements a light-weight mixture-of-experts (MoE) block that can be
plugged into U-Net skip connections. The implementation follows the ideas from
the UTANet paper, but is simplified so it can be re-used across different
architectures in this repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A simple expert made of point-wise convolutions."""

    def __init__(self, emb_size: int, hidden_rate: int = 2) -> None:
        super().__init__()
        hidden = hidden_rate * emb_size
        self.net = nn.Sequential(
            nn.Conv2d(emb_size, hidden, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, emb_size, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoE(nn.Module):
    """
    Mixture-of-Experts router that selects top-k experts for every sample.

    Args:
        emb_size: number of channels of the input embedding.
        num_experts: how many experts to maintain.
        top_k: number of experts selected per sample.
        hidden_rate: expert width multiplier.
    """

    def __init__(
        self,
        emb_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_rate: int = 2,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts.")

        self.emb_size = emb_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [Expert(emb_size, hidden_rate=hidden_rate) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(emb_size, num_experts)
        self.gap = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        """Coefficient of variation squared used for load-balancing."""
        eps = 1e-10
        if x.numel() <= 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return x.float().var(unbiased=False) / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor shaped (B, C, H, W) where C == emb_size

        Returns:
            processed tensor with same shape as input
            auxiliary load-balancing loss
        """
        if x.shape[1] != self.emb_size:
            raise ValueError(
                f"Expected {self.emb_size} channels, but got {x.shape[1]} channels."
            )

        b, _, _, _ = x.shape
        gap = self.gap(x).view(b, -1)
        gate_logits = self.gate(gap)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        # keep only top-k experts per sample
        top_vals, top_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, top_idx, 1.0)
        gated = gate_probs * mask
        gated = gated / (gated.sum(dim=-1, keepdim=True) + 1e-8)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, E, C, H, W)

        weights = gated.view(b, self.num_experts, 1, 1, 1)
        out = (expert_stack * weights).sum(dim=1)

        expert_usage = gated.sum(dim=0)  # (E,)
        aux_loss = self.cv_squared(expert_usage)

        return out, aux_loss


class SkipMoSCLayer(nn.Module):
    """
    Applies a shared MoE router to skip features.

    Each skip feature is projected to the shared embedding dimension, processed
    by the shared MoE, and then projected back to its original channel count.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.to_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.from_embed = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, skip: torch.Tensor, moe: MoE) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.to_embed(skip)
        routed, aux_loss = moe(emb)
        refined = self.from_embed(routed)
        return refined, aux_loss


__all__ = ["MoE", "SkipMoSCLayer"]
