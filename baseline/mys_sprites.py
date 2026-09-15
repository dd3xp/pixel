"""External baseline: big text-to-image render + Make Your Own Sprites (Wu et al., SIGGRAPH Asia 2022, TOG 41(6);
github.com/WuZongWei6/Pixelization), the academic state of the art in learned image-to-pixel-art conversion.

Same protocol as pixeloe_sprites.py: cut the sprite out of its plain background (api_gen.cutout), pad to a square on
white, pixelize the RGB, get alpha by majority vote of the cutout mask inside each cell, zero RGB under transparency.
The network paints 4-px cells at an input of (cells x 4) px, so the square is resized straight to T*4 (BICUBIC, as in
the official pixelize) and the output is NEAREST-reduced by 4 to exactly T x T. The official test_pro.py first forces
images to >= 128 px, which for 12-24 px targets would change the cell count, so that step is bypassed.
Usage (GPU; the repo's define_G requires CUDA):
  python baseline/mys_sprites.py --repo ../Pixelization --model mys --big runs_out/ext200/runs/ext_recap_gpt/big \
         --out runs_out/ext200/runs/ext_recap_gpt_mys --size 16
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_gen import cutout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", default="mys", help="checkpoints/<model>/160_net_G_A.pth inside the repo")
    ap.add_argument("--big", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, nargs="+", default=[16])
    a = ap.parse_args()
    big = Path(a.big).resolve()
    outs = {T: Path(a.out).resolve() / f"s{T}" for T in a.size}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    os.chdir(a.repo)                                   # the repo loads ./alias_net.pth and ./checkpoints/...
    sys.path.insert(0, os.getcwd())
    import test_pro
    m = test_pro.Model(a.model, device="cuda")
    m.load()
    files = sorted(big.glob("*.png")) + sorted(big.glob("*.jpg"))
    n = 0
    with torch.no_grad():
        for f in files:
            sprite = cutout(Image.open(f))
            rgba = np.asarray(sprite.convert("RGBA"))
            h, w = rgba.shape[:2]
            side = max(h, w)
            sq_rgb = np.full((side, side, 3), 255, np.uint8)
            sq_a = np.zeros((side, side), np.uint8)
            y0, x0 = (side - h) // 2, (side - w) // 2
            sq_rgb[y0:y0 + h, x0:x0 + w] = rgba[..., :3]
            sq_a[y0:y0 + h, x0:x0 + w] = rgba[..., 3]
            for T, o in outs.items():
                img = Image.fromarray(sq_rgb).resize((T * 4, T * 4), Image.BICUBIC)
                t = test_pro.process(img).to(m.device)
                y = m.alias_net(m.G_A_net.module.RGBDec(m.G_A_net.module.RGBEnc(t), m.cell_size_code))
                arr = ((y[0].permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                rgb = np.asarray(Image.fromarray(arr).resize((T, T), Image.NEAREST))
                alpha = (np.asarray(Image.fromarray(sq_a).resize((T, T), Image.BOX)) >= 128).astype(np.uint8) * 255
                out = np.dstack([rgb, alpha])
                out[alpha == 0] = 0
                Image.fromarray(out, "RGBA").save(o / f"{f.stem}.png")
            n += 1
    print(f"MYS: {n} sprites x sizes {a.size} -> {Path(a.out)}", flush=True)


if __name__ == "__main__":
    main()
