"""Diagnostics for detail preservation — the P0 measurement kit.

These quantify the failure mode observed in EXP-002 (SDS erasing highlights,
tonal levels, and shading), so every later phase can report the same curves.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def colors_used(indices_hw: torch.Tensor) -> int:
    """Number of distinct palette entries actually used."""
    return int(indices_hw.unique().numel())


@torch.no_grad()
def l1_to_reference(image_3hw: torch.Tensor, reference_3hw: torch.Tensor) -> float:
    """Mean L1 drift from a reference image of the same size."""
    return float((image_3hw - reference_3hw).abs().mean())


@torch.no_grad()
def highfreq_energy(image_3hw: torch.Tensor) -> float:
    """Mean absolute Laplacian response — proxy for shading/dither detail."""
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=image_3hw.device)
    k = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    resp = F.conv2d(image_3hw.unsqueeze(0), k, groups=3, padding=1)
    return float(resp.abs().mean())


@torch.no_grad()
def luminance_levels(image_3hw: torch.Tensor, bins: int = 8) -> int:
    """Number of occupied luminance bins — proxy for tonal range survival."""
    lum = (image_3hw * torch.tensor([0.299, 0.587, 0.114], device=image_3hw.device).view(3, 1, 1)).sum(0)
    hist = torch.histc(lum, bins=bins, min=0.0, max=1.0)
    return int((hist > 0).sum())
