"""Offline reference-chain preview: build per-level reference images from a config
WITHOUT loading any diffusion model. Seconds, not minutes — validate preprocessing
and downsampling operators here BEFORE spending a GPU run.

Usage: python scripts/preview_ref.py -c configs/foo.yaml -o workdir/preview_foo
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimize import (_load_image, _load_mask, _resize, _tone_boost,  # noqa: E402
                          _valley_expand, _gap_tophat, _save_png)
from src.palette import load_hex_palette, pixel_palette_distances  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or f"workdir/preview_{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    source = _load_image(cfg["image"], device)
    mask = _load_mask(cfg["mask"], device) if cfg.get("mask") else None
    palette = load_hex_palette(cfg["palette"]).to(device)

    if cfg.get("tone_gain"):
        tg = cfg["tone_gain"]
        source = _tone_boost(source, mask, gain=tg if isinstance(tg, list) else float(tg),
                             unsharp=float(cfg.get("tone_unsharp", 0.5)))
    if cfg.get("gap_gain"):
        import torch.nn.functional as F
        bg_col = None
        gmask = mask
        if mask is not None:
            bg_px = source[:, mask.squeeze(0) < 0.5].T
            bg_col = bg_px.median(dim=0).values
            gme = int(cfg.get("gap_mask_erode", 0))
            if gme > 0:
                gmask = -F.max_pool2d(-mask.unsqueeze(0), gme * 2 + 1, stride=1, padding=gme).squeeze(0)
        source = _gap_tophat(source, gmask, kernel=int(cfg.get("gap_kernel", 41)),
                             gain=float(cfg["gap_gain"]), cap=float(cfg.get("gap_cap", 0.8)),
                             target_color=bg_col)
    elif cfg.get("valley_gain"):
        source = _valley_expand(source, mask, kernel=int(cfg.get("valley_kernel", 33)),
                                gain=float(cfg["valley_gain"]))
    _save_png(source, out / "preprocessed_source.png", resize=512)

    ds_mode = cfg.get("downsample", "bilinear")
    for size in [int(s) for s in cfg.get("scales", [64, 32, 24, 16])]:
        ref = _resize(source, size, mode=ds_mode)
        _save_png(ref, out / f"ref_{size}.png")
        # palette-projected view = what the anchor/init actually sees
        d = pixel_palette_distances(ref.permute(1, 2, 0), palette)
        proj = palette[d.argmin(-1)].permute(2, 0, 1)
        _save_png(proj, out / f"ref_{size}_projected.png")
    print(f"previews -> {out}")


if __name__ == "__main__":
    main()
