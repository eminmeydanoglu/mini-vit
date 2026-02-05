"""
Perceptual Loss using pretrained VGG16.
Compares images in feature space, not pixel space.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """
    Extract features from pretrained VGG16.
    Uses early-mid layers (before fully connected).
    """

    def __init__(self, layer_idx: int = 16):
        """
        Args:
            layer_idx: Which layer to extract features from.
                       VGG16 features structure:
                       0-3: conv1 (64 filters)
                       4-8: conv2 (128 filters)
                       9-15: conv3 (256 filters)
                       16-22: conv4 (512 filters) <- good balance
                       23-29: conv5 (512 filters)
        """
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:layer_idx])

        # Freeze - we don't train VGG
        for param in self.parameters():
            param.requires_grad = False

        # VGG expects ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) images in [0, 1] range
        Returns:
            (B, C, H', W') feature maps
        """
        # Normalize to ImageNet stats
        x = (x - self.mean) / self.std
        return self.features(x)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss = MSE between VGG features of pred and target.
    """

    def __init__(self, layer_idx: int = 16):
        super().__init__()
        self.vgg = VGGFeatureExtractor(layer_idx)
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) predicted images
            target: (B, 3, H, W) target images
        Returns:
            Scalar loss
        """
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.criterion(pred_features, target_features)


class CombinedLoss(nn.Module):
    """
    Combined loss = MSE + λ * Perceptual
    """

    def __init__(self, lambda_perceptual: float = 0.1, layer_idx: int = 16):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.perceptual = PerceptualLoss(layer_idx)
        self.lambda_p = lambda_perceptual

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: predicted images
            target: target images
            mask: (B, N) patch mask for masked MSE (optional)
        Returns:
            total_loss, mse_loss, perceptual_loss
        """
        # MSE loss (masked if mask provided)
        if mask is not None:
            B, C, H, W = target.shape
            P = 4  # patch_size
            grid = H // P

            mask_2d = mask.reshape(B, grid, grid).unsqueeze(1).float()
            mask_full = mask_2d.repeat_interleave(P, dim=2).repeat_interleave(P, dim=3)
            mask_full = mask_full.expand(-1, C, -1, -1)

            pixel_loss = self.mse(pred, target)
            mse_loss = (pixel_loss * mask_full).sum() / mask_full.sum()
        else:
            mse_loss = self.mse(pred, target).mean()

        # Perceptual loss (on full image)
        perceptual_loss = self.perceptual(pred, target)

        # Combined
        total_loss = mse_loss + self.lambda_p * perceptual_loss

        return total_loss, mse_loss, perceptual_loss


if __name__ == "__main__":
    # Test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = CombinedLoss(lambda_perceptual=0.1).to(device)

    # Dummy images
    pred = torch.rand(4, 3, 32, 32).to(device)
    target = torch.rand(4, 3, 32, 32).to(device)
    mask = torch.randint(0, 2, (4, 64)).bool().to(device)

    total, mse, perc = loss_fn(pred, target, mask)

    print(f"MSE Loss: {mse.item():.4f}")
    print(f"Perceptual Loss: {perc.item():.4f}")
    print(f"Total Loss: {total.item():.4f}")
    print("Test passed!")
