"""
Training with Perceptual Loss.
MSE + λ * VGG Feature Loss for sharper reconstructions.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from src.model import create_mini_vit
from src.dataset import MaskedCIFAR10
from src.perceptual import CombinedLoss

# Configuration
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
EPOCHS = 20
LAMBDA_PERCEPTUAL = 0.1  # Weight for perceptual loss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints_perceptual"


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Lambda Perceptual: {LAMBDA_PERCEPTUAL}")

    # Dataset
    print("Loading dataset...")
    train_dataset = MaskedCIFAR10(train=True, mask_ratio=0.5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Model (fresh, not pretrained)
    print("Creating Mini-ViT...")
    model = create_mini_vit(embed_dim=192, depth=6, num_heads=6, dropout=0.1).to(DEVICE)

    # Combined Loss
    criterion = CombinedLoss(lambda_perceptual=LAMBDA_PERCEPTUAL).to(DEVICE)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
    )

    # Training
    print("Starting training with Perceptual Loss...")
    history = {"total": [], "mse": [], "perceptual": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_mse = 0
        total_perc = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for _, original, mask in pbar:
            original = original.to(DEVICE)
            mask = mask.to(DEVICE)

            reconstructed = model(original, mask=mask)

            # Clamp to valid range for VGG
            reconstructed_clamped = reconstructed.clamp(0, 1)

            loss, mse_loss, perc_loss = criterion(reconstructed_clamped, original, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_perc += perc_loss.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{mse_loss.item():.4f}",
                    "perc": f"{perc_loss.item():.4f}",
                }
            )

        n = len(train_loader)
        history["total"].append(total_loss / n)
        history["mse"].append(total_mse / n)
        history["perceptual"].append(total_perc / n)

        print(
            f"Epoch {epoch + 1}: Total={total_loss / n:.4f}, MSE={total_mse / n:.4f}, Perc={total_perc / n:.4f}"
        )

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(
                SAVE_DIR, f"mini_vit_perceptual_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), save_path)
            visualize_comparison(model, train_dataset, epoch + 1)

    # Save loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["total"], label="Total")
    plt.plot(history["mse"], label="MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["perceptual"], label="Perceptual", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Perceptual Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curves.png"))
    plt.close()


def visualize_comparison(model, dataset, epoch):
    """Visualize reconstructions."""
    model.eval()

    indices = torch.randperm(len(dataset))[:4]

    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    plt.suptitle(f"Perceptual Loss - Epoch {epoch}")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            _, original, mask = dataset[idx]

            input_tensor = original.unsqueeze(0).to(DEVICE)
            mask_tensor = mask.unsqueeze(0).to(DEVICE)

            reconstruction = model(input_tensor, mask=mask_tensor)

            # Original
            orig_img = original.permute(1, 2, 0).numpy()
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            # Masked input
            masked_vis = dataset._apply_mask(original, mask)
            masked_vis = masked_vis.permute(1, 2, 0).numpy()
            axes[i, 1].imshow(masked_vis)
            axes[i, 1].set_title("Masked")
            axes[i, 1].axis("off")

            # Reconstruction
            recon_img = reconstruction.squeeze(0).cpu().permute(1, 2, 0).numpy()
            recon_img = recon_img.clip(0, 1)
            axes[i, 2].imshow(recon_img)
            axes[i, 2].set_title("Reconstruction")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"viz_epoch_{epoch}.png"))
    plt.close()


if __name__ == "__main__":
    train()
