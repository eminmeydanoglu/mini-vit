"""Attention visualization for Mini-ViT."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torchvision import datasets, transforms

from src.model import MiniViT, ViTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AttentionExtractor:
    """Hook-based attention extractor."""

    def __init__(self, model: MiniViT):
        self.model = model
        self.attentions: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        for block in self.model.blocks:
            hook = block.attn.register_forward_hook(self._attention_hook)
            self._hooks.append(hook)

    def _attention_hook(self, module, input, output):
        x = input[0]
        B, N, D = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)

        self.attentions.append(attn.detach().cpu())

    def clear(self):
        self.attentions = []

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.clear()
        with torch.no_grad():
            self.model(x)
        return self.attentions


def visualize_attention(
    model: MiniViT,
    image: torch.Tensor,
    query_patch: int = 32,
    save_path: str = "attention_maps.png",
):
    """Visualize attention from a query patch to all other patches."""
    model.eval()
    extractor = AttentionExtractor(model)

    x = image.unsqueeze(0).to(DEVICE)
    attentions = extractor(x)
    extractor.remove_hooks()

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    grid_size = 8

    fig, axes = plt.subplots(
        num_layers, num_heads + 1, figsize=(2 * (num_heads + 1), 2 * num_layers)
    )

    img_np = image.permute(1, 2, 0).numpy()
    for layer_idx in range(num_layers):
        axes[layer_idx, 0].imshow(img_np)
        axes[layer_idx, 0].set_title(f"Layer {layer_idx + 1}" if layer_idx == 0 else "")
        axes[layer_idx, 0].axis("off")

        query_row = query_patch // grid_size
        query_col = query_patch % grid_size
        rect = mpatches.Rectangle(
            (query_col * 4, query_row * 4),
            4,
            4,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axes[layer_idx, 0].add_patch(rect)

    for layer_idx, attn in enumerate(attentions):
        attn = attn[0]

        for head_idx in range(num_heads):
            attn_map = attn[head_idx, query_patch, :]
            attn_map = attn_map.reshape(grid_size, grid_size).numpy()

            ax = axes[layer_idx, head_idx + 1]
            ax.imshow(attn_map, cmap="viridis", vmin=0, vmax=attn_map.max())

            if layer_idx == 0:
                ax.set_title(f"Head {head_idx + 1}")
            ax.axis("off")

    plt.suptitle(f"Attention from patch {query_patch} (red box)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def visualize_mean_attention_distance(
    model: MiniViT, images: torch.Tensor, save_path: str = "attention_distance.png"
):
    """Compute mean attention distance per layer/head (cf. ViT paper Figure 7)."""
    model.eval()
    extractor = AttentionExtractor(model)

    x = images.to(DEVICE)
    attentions = extractor(x)
    extractor.remove_hooks()

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    grid_size = 8

    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            positions.append((i * 4 + 2, j * 4 + 2))
    positions = torch.tensor(positions).float()

    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    distances = torch.sqrt((diff**2).sum(-1))

    mean_distances = np.zeros((num_layers, num_heads))

    for layer_idx, attn in enumerate(attentions):
        attn = attn.mean(0)
        for head_idx in range(num_heads):
            attn_head = attn[head_idx]
            weighted_dist = (attn_head * distances).sum() / attn_head.sum()
            mean_distances[layer_idx, head_idx] = weighted_dist.item()

    fig, ax = plt.subplots(figsize=(10, 6))

    for head_idx in range(num_heads):
        ax.plot(
            range(1, num_layers + 1),
            mean_distances[:, head_idx],
            "o-",
            label=f"Head {head_idx + 1}",
            alpha=0.7,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Distance (pixels)")
    ax.set_title("Attention Distance by Layer and Head")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Loading model...")
    config = ViTConfig()
    model = MiniViT(config)
    model.load_state_dict(
        torch.load("checkpoints/mini_vit_epoch_20.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root="./data", train=False, transform=transform)

    test_indices = [49, 23, 15, 8]

    for idx in test_indices:
        image, label = dataset[idx]
        class_name = dataset.classes[label]
        print(f"Visualizing: {class_name} (index {idx})")
        visualize_attention(
            model,
            image,
            query_patch=27,
            save_path=f"attention_{class_name}_{idx}.png",
        )

    print("Computing mean attention distance...")
    images = torch.stack([dataset[i][0] for i in range(100)])
    visualize_mean_attention_distance(model, images, "attention_distance.png")
    print("Done!")


if __name__ == "__main__":
    main()
