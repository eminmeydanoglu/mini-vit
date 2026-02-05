"""Training script for Mini-ViT masked patch prediction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MaskedCIFAR10
from src.model import create_mini_vit

BATCH_SIZE = 128
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 0.05
EPOCHS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints"


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    train_dataset = MaskedCIFAR10(train=True, mask_ratio=0.5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = create_mini_vit(embed_dim=192, depth=6, num_heads=6, dropout=0.1).to(DEVICE)

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

    criterion = nn.MSELoss(reduction="none")
    history = {"loss": []}

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for _, original, mask in pbar:
            original = original.to(DEVICE)
            mask = mask.to(DEVICE)

            reconstructed = model(original, mask=mask)

            # Compute loss only on masked patches
            B, C, H, W = original.shape
            P = 4
            grid = H // P

            mask_2d = mask.reshape(B, grid, grid).unsqueeze(1).float()
            mask_full = mask_2d.repeat_interleave(P, dim=2).repeat_interleave(P, dim=3)
            mask_full = mask_full.expand(-1, C, -1, -1)

            pixel_loss = criterion(reconstructed, original)
            masked_loss = (pixel_loss * mask_full).sum() / mask_full.sum()

            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += masked_loss.item()
            pbar.set_postfix({"loss": masked_loss.item()})

        avg_loss = total_loss / len(train_loader)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(SAVE_DIR, f"mini_vit_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            visualize_progress(model, train_dataset, epoch + 1)


def visualize_progress(model, dataset, epoch):
    """Visualize current model predictions."""
    model.eval()
    indices = torch.randperm(len(dataset))[:4]

    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    plt.suptitle(f"Epoch {epoch} Predictions")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            _, original, mask = dataset[idx]
            input_tensor = original.unsqueeze(0).to(DEVICE)
            mask_tensor = mask.unsqueeze(0).to(DEVICE)

            reconstruction = model(input_tensor, mask=mask_tensor)

            orig_img = original.permute(1, 2, 0).numpy()
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            masked_vis = dataset._apply_mask(original, mask)
            masked_vis = masked_vis.permute(1, 2, 0).numpy()
            axes[i, 1].imshow(masked_vis)
            axes[i, 1].set_title("Input (Masked)")
            axes[i, 1].axis("off")

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
