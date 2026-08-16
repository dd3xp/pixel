"""Anchor losses — the P2 research surface.

P0 ships one working stub (low-pass L1) wired into the loop with weight 0 by
default, so the plumbing exists and P2 can iterate on variants without touching
the optimizer loop.
"""
import torch
import torch.nn.functional as F


def _gaussian_blur(img_3hw: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=img_3hw.device, dtype=img_3hw.dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = (g / g.sum()).view(1, 1, -1)
    img = img_3hw.unsqueeze(0)
    img = F.conv2d(img, g.unsqueeze(2).repeat(3, 1, 1, 1), groups=3, padding=(0, radius))
    img = F.conv2d(img, g.unsqueeze(3).repeat(3, 1, 1, 1), groups=3, padding=(radius, 0))
    return img.squeeze(0)


def lowpass_l1_anchor(image_3hw: torch.Tensor, reference_3hw: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """L1 between low-pass versions: anchors layout/colors, leaves detail free."""
    return (_gaussian_blur(image_3hw, sigma) - _gaussian_blur(reference_3hw, sigma)).abs().mean()
