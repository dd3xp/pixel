"""No-SDS control: stylized source -> crop -> outline-aware downscale -> palette
-> cleanup. Pure deterministic chain, seconds, no optimization. If this beats the
pipeline output, the SDS stages are net-negative post-stylization.

Usage: python scripts/direct_chain.py <stylized.png> <out_dir> [palette]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimize import _auto_crop, _load_image, _resize, _save_png  # noqa: E402
from src.palette import load_hex_palette, pixel_palette_distances  # noqa: E402


def project(img_3hw: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    d = pixel_palette_distances(img_3hw.permute(1, 2, 0), palette)
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


def main() -> None:
    src_path, out_dir = sys.argv[1], Path(sys.argv[2])
    pal_path = sys.argv[3] if len(sys.argv) > 3 else "assets/palettes/dawnbringer32.hex"
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = load_hex_palette(pal_path)
    src = _load_image(src_path, "cpu")
    src, _ = _auto_crop(src, None)
    for size in [32, 24, 16]:
        r = _resize(src, size, mode="kcentroid_outline")
        r = project(r, palette)
        r = despeckle(r, palette)
        _save_png(r, out_dir / f"direct_{size}.png")
    print(f"direct chain -> {out_dir}")


if __name__ == "__main__":
    main()
