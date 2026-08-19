"""No-SDS control: stylized source -> crop -> outline-aware downscale -> palette
-> cleanup. Pure deterministic chain, seconds, no optimization. If this beats the
pipeline output, the SDS stages are net-negative post-stylization.

Usage: python scripts/direct_chain.py <stylized.png> <out_dir> [palette]
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimize import _auto_crop, _load_image, _resize, _save_png  # noqa: E402
from src.palette import load_hex_palette, pixel_palette_distances  # noqa: E402


def project(img_3hw: torch.Tensor, palette: torch.Tensor, hue_weighted: bool = True) -> torch.Tensor:
    if not hue_weighted:
        d = pixel_palette_distances(img_3hw.permute(1, 2, 0), palette)
        return palette[d.argmin(-1)].permute(2, 0, 1)
    # hue-weighted projection: match chromaticity first, luminance second — a dark
    # red maps to the palette RED (not to a luminance-matched brown)
    px = img_3hw.permute(1, 2, 0)  # (H, W, 3)
    eps = 1e-4
    pc = px / (px.sum(-1, keepdim=True) + eps)          # chromaticity
    qc = palette / (palette.sum(-1, keepdim=True) + eps)
    d_chroma = (pc.unsqueeze(-2) - qc).abs().sum(-1)     # (H, W, K)
    w = torch.tensor([0.299, 0.587, 0.114])
    d_lum = ((px @ w).unsqueeze(-1) - (palette @ w)).abs()
    d = 2.0 * d_chroma + 0.8 * d_lum
    return palette[d.argmin(-1)].permute(2, 0, 1)


def despeckle(img_3hw: torch.Tensor, palette: torch.Tensor, thresh: float = 0.9) -> torch.Tensor:
    d = pixel_palette_distances(img_3hw.permute(1, 2, 0), palette)
    idx = d.argmin(-1)
    H, W = idx.shape
    out_idx = idx.clone()
    for y in range(H):
        for x in range(W):
            neigh = [idx[yy, xx] for yy in range(max(0, y - 1), min(H, y + 2))
                     for xx in range(max(0, x - 1), min(W, x + 2)) if (yy, xx) != (y, x)]
            nt = torch.stack(neigh)
            if (nt != idx[y, x]).all():
                own = palette[idx[y, x]]
                if (palette[nt] - own).abs().sum(-1).min() > thresh:
                    out_idx[y, x] = nt.mode().values
    return palette[out_idx].permute(2, 0, 1)


def flatten_bg(img_3hw: torch.Tensor, thresh: float = 0.4) -> torch.Tensor:
    """Flood-fill flatten: pixels near the border color AND connected to the
    border become one flat bg color. Subject pixels of similar color (e.g. a
    cream stem on beige bg) survive because they are not border-connected...
    unless they touch the silhouette edge - grow is color-gated."""
    C, H, W = img_3hw.shape
    border = torch.cat([img_3hw[:, 0, :], img_3hw[:, -1, :], img_3hw[:, :, 0], img_3hw[:, :, -1]], dim=1)
    med = border.median(dim=1).values
    near = (img_3hw - med.view(3, 1, 1)).abs().sum(0) < thresh
    bg = torch.zeros_like(near)
    bg[0, :], bg[-1, :], bg[:, 0], bg[:, -1] = near[0, :], near[-1, :], near[:, 0], near[:, -1]
    for _ in range(H + W):
        grown = (F.max_pool2d(bg.float()[None, None], 3, 1, 1)[0, 0] > 0) & near
        if (grown == bg).all():
            break
        bg = grown
    out = img_3hw.clone()
    out[:, bg] = med.view(3, 1).expand(3, int(bg.sum()))
    return out


def main() -> None:
    src_path, out_dir = sys.argv[1], Path(sys.argv[2])
    pal_path = sys.argv[3] if len(sys.argv) > 3 else "assets/palettes/dawnbringer32.hex"
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = load_hex_palette(pal_path)
    src = _load_image(src_path, "cpu")
    src, _ = _auto_crop(src, None)
    for size in [32, 24, 16]:
        r = _resize(src, size, mode="kcentroid_outline")
        r = flatten_bg(r)
        r = project(r, palette)
        r = despeckle(r, palette)
        _save_png(r, out_dir / f"direct_{size}.png")
    print(f"direct chain -> {out_dir}")


if __name__ == "__main__":
    main()
