"""Linear probe evaluation for learned representations."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model import MiniViT, ViTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3


class LinearProbe(nn.Module):
    """Frozen encoder + trainable linear classifier."""

    def __init__(self, encoder: MiniViT, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(encoder.config.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed

        for block in self.encoder.blocks:
            x = block(x)

        x = self.encoder.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x


def load_pretrained_encoder(checkpoint_path: str) -> MiniViT:
    config = ViTConfig()
    model = MiniViT(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model


def train_linear_probe(checkpoint_path: str) -> float:
    print(f"Device: {DEVICE}")
    print(f"Loading encoder from {checkpoint_path}...")

    encoder = load_pretrained_encoder(checkpoint_path)
    probe = LinearProbe(encoder).to(DEVICE)

    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    total = sum(p.numel() for p in probe.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    optimizer = optim.Adam(probe.classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Training linear probe...")
    test_acc = 0.0

    for epoch in range(EPOCHS):
        probe.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = probe(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        train_acc = 100.0 * correct / total

        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = probe(images)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * correct / total
        print(
            f"Epoch {epoch + 1}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%"
        )

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    return test_acc


if __name__ == "__main__":
    train_linear_probe("checkpoints/mini_vit_epoch_20.pth")
