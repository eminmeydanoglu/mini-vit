"""
Mini-ViT for Masked Patch Prediction on CIFAR-10.
Architecture follows Dosovitskiy et al., 2020 (ViT paper).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ViTConfig:
    """Mini-ViT configuration for CIFAR-10."""

    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.patch_size**2 * self.in_channels


class PatchEmbedding(nn.Module):
    """Image to patch embedding via Conv2d (equivalent to linear projection)."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (MSA)."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with Pre-LayerNorm."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViT(nn.Module):
    """Mini Vision Transformer for Masked Patch Prediction."""

    def __init__(self, config: ViTConfig | None = None):
        super().__init__()
        self.config = config or ViTConfig()

        self.patch_embed = PatchEmbedding(self.config)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.config.num_patches, self.config.embed_dim)
        )
        self.pos_dropout = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.depth)]
        )
        self.norm = nn.LayerNorm(self.config.embed_dim)

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))

        # Reconstruction head: embed_dim -> patch pixels
        self.reconstruction_head = nn.Linear(
            self.config.embed_dim, self.config.patch_dim
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image
            mask: (B, N) boolean mask. True = masked patch.
        Returns:
            (B, C, H, W) reconstructed image
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        if mask is not None:
            w = mask.unsqueeze(-1).type_as(x)
            mask_tokens = self.mask_token.expand(B, x.shape[1], -1)
            x = x * (1 - w) + mask_tokens * w

        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.reconstruction_head(x)
        x = self._patches_to_image(x)

        return x

    def _patches_to_image(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, N, P*P*C) patch predictions to (B, C, H, W) image."""
        B = x.shape[0]
        P = self.config.patch_size
        C = self.config.in_channels
        H = W = self.config.image_size
        grid_size = H // P

        x = x.reshape(B, grid_size, grid_size, P, P, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mini_vit(
    embed_dim: int = 192,
    depth: int = 6,
    num_heads: int = 6,
    dropout: float = 0.1,
) -> MiniViT:
    """Factory function for Mini-ViT with custom config."""
    config = ViTConfig(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        dropout=dropout,
    )
    return MiniViT(config)


if __name__ == "__main__":
    config = ViTConfig()
    model = MiniViT(config)

    print(
        f"Config: {config.image_size}x{config.image_size}, {config.num_patches} patches"
    )
    print(
        f"Embed dim: {config.embed_dim}, Depth: {config.depth}, Heads: {config.num_heads}"
    )
    print(f"Parameters: {model.count_parameters():,}")

    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
