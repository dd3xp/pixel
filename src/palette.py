"""Palette loading and color utilities."""
from pathlib import Path

import torch


def load_hex_palette(path: str | Path) -> torch.Tensor:
    """Load a lospec-style .hex file (one RRGGBB per line) -> (K, 3) float tensor in [0, 1]."""
    colors = []
    for line in Path(path).read_text().splitlines():
        line = line.strip().lstrip("#")
        if len(line) == 6:
            colors.append([int(line[i : i + 2], 16) for i in (0, 2, 4)])
    if not colors:
        raise ValueError(f"No colors parsed from {path}")
    return torch.tensor(colors, dtype=torch.float32) / 255.0


def pixel_palette_distances(image_hw3: torch.Tensor, palette_k3: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    """Distance of every pixel to every palette color. image (H, W, 3), palette (K, 3) -> (H, W, K)."""
    diff = image_hw3.unsqueeze(-2) - palette_k3  # (H, W, K, 3)
    if mode == "l1":
        return diff.abs().sum(-1)
    if mode == "l2":
        return diff.pow(2).sum(-1).sqrt()
    raise ValueError(f"Unknown distance mode: {mode}")
