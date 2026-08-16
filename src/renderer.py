"""Differentiable palette renderer.

The image is represented as per-pixel logits over the palette; rendering is a
(gumbel-)softmax-weighted combination of palette colors. Hard (argmax) rendering
gives the final quantized pixel art.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .palette import pixel_palette_distances


class PaletteRenderer(nn.Module):
    def __init__(self, height: int, width: int, palette: torch.Tensor, init_std: float = 1.0):
        super().__init__()
        self.height, self.width = height, width
        self.register_buffer("palette", palette)  # (K, 3)
        self.logits = nn.Parameter(torch.randn(height, width, palette.shape[0]) * init_std)

    @property
    def num_colors(self) -> int:
        return self.palette.shape[0]

    def init_from_image(self, image_3hw: torch.Tensor, distance: str = "l1", sharpness: float = 10.0) -> None:
        """Initialize logits from an image already resized to (H, W).

        Logits are set to -sharpness * distance(pixel, color), so softmax rendering
        starts near the palette-projected image (SD-piXL's 'palette-bilinear' init).
        """
        image_hw3 = image_3hw.permute(1, 2, 0).to(self.logits.device)
        d = pixel_palette_distances(image_hw3, self.palette, mode=distance)  # (H, W, K)
        with torch.no_grad():
            self.logits.copy_(-sharpness * d)

    def forward(self, tau: float = 1.0, mode: str = "gumbel") -> torch.Tensor:
        """Render to (3, H, W) in [0, 1]. mode: 'softmax' | 'gumbel' | 'hard'."""
        if mode == "hard":
            idx = self.logits.argmax(-1)
            img = self.palette[idx]  # (H, W, 3)
        elif mode == "gumbel":
            w = F.gumbel_softmax(self.logits, tau=tau, hard=True, dim=-1)  # straight-through
            img = w @ self.palette
        elif mode == "softmax":
            w = F.softmax(self.logits / tau, dim=-1)
            img = w @ self.palette
        else:
            raise ValueError(f"Unknown render mode: {mode}")
        return img.permute(2, 0, 1)

    @torch.no_grad()
    def hard_indices(self) -> torch.Tensor:
        return self.logits.argmax(-1)  # (H, W)
