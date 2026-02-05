"""CIFAR-10 dataset with random patch masking for masked patch prediction."""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MaskedCIFAR10(Dataset):
    """CIFAR-10 with random patch masking."""

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        patch_size: int = 4,
        mask_ratio: float = 0.5,
    ):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.image_size = 32

        self.grid_size = self.image_size // patch_size
        self.num_patches = self.grid_size**2
        self.num_masked = int(self.num_patches * mask_ratio)

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=self.transform
        )
        self.classes = self.dataset.classes

    def _create_mask(self) -> torch.Tensor:
        """Create random patch mask. Returns (num_patches,) bool tensor."""
        indices = torch.randperm(self.num_patches)[: self.num_masked]
        mask = torch.zeros(self.num_patches, dtype=torch.bool)
        mask[indices] = True
        return mask

    def _apply_mask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero out masked patches in image."""
        masked = image.clone()
        mask_2d = mask.reshape(self.grid_size, self.grid_size)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if mask_2d[i, j]:
                    y_start = i * self.patch_size
                    y_end = y_start + self.patch_size
                    x_start = j * self.patch_size
                    x_end = x_start + self.patch_size
                    masked[:, y_start:y_end, x_start:x_end] = 0.0

        return masked

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original, _ = self.dataset[idx]
        mask = self._create_mask()
        masked = self._apply_mask(original, mask)
        return masked, original, mask


def visualize_samples(dataset: MaskedCIFAR10, n_samples: int = 4):
    """Visualize dataset samples."""
    fig, axes = plt.subplots(n_samples, 2, figsize=(6, 3 * n_samples))

    for i in range(n_samples):
        masked, original, mask = dataset[i]

        masked_np = masked.permute(1, 2, 0).numpy()
        original_np = original.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(original_np)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masked_np)
        axes[i, 1].set_title(f"Masked ({dataset.num_masked}/{dataset.num_patches})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("masked_samples.png", dpi=150)
    plt.show()
    print("Saved: masked_samples.png")


if __name__ == "__main__":
    print("Loading CIFAR-10...")
    dataset = MaskedCIFAR10(train=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Patch size: {dataset.patch_size}x{dataset.patch_size}")
    print(
        f"Grid: {dataset.grid_size}x{dataset.grid_size} = {dataset.num_patches} patches"
    )
    print(f"Masked: {dataset.num_masked} patches ({dataset.mask_ratio * 100:.0f}%)")

    masked, original, mask = dataset[0]
    print(
        f"\nSample shapes: masked={masked.shape}, original={original.shape}, mask={mask.shape}"
    )

    visualize_samples(dataset, n_samples=4)
