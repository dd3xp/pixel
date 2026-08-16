"""Single-scale optimization loop (P0). P1 extends this into the multi-scale pyramid."""
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .losses import lowpass_l1_anchor
from .metrics import colors_used, highfreq_energy, l1_to_reference, luminance_levels
from .palette import load_hex_palette
from .renderer import PaletteRenderer
from .sds import SDXLGuidance


def _load_image(path: str, size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(img.height, img.width, 3) / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return F.interpolate(t, size=(size, size), mode="bilinear", antialias=True).squeeze(0)


def _save_png(image_3hw: torch.Tensor, path: Path, resize: int | None = 256) -> None:
    arr = (image_3hw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    if resize and resize > img.width:
        img = img.resize((resize, resize), Image.NEAREST)
    img.save(path)


def run(cfg: dict, out_dir: Path) -> None:
    device = cfg.get("device", "cuda")
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    png_dir = out_dir / "png_logs"
    png_dir.mkdir(parents=True, exist_ok=True)

    palette = load_hex_palette(cfg["palette"]).to(device)
    H = W = int(cfg["image_size"])
    renderer = PaletteRenderer(H, W, palette, init_std=cfg.get("init_std", 1.0)).to(device)

    reference_small = None
    if cfg.get("image"):
        reference_small = _load_image(cfg["image"], H, device)
        renderer.init_from_image(reference_small, distance=cfg.get("init_distance", "l1"))

    guidance = SDXLGuidance(
        model_id=cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
        device=device,
        render_size=int(cfg.get("render_size", 1024)),
    )
    guidance.set_prompt(cfg["prompt"], cfg.get("negative_prompt", ""))

    opt = torch.optim.Adam(renderer.parameters(), lr=float(cfg.get("lr", 0.025)))
    steps = int(cfg["steps"])
    save_steps = int(cfg.get("save_steps", 50))
    tau_min, tau_max = cfg.get("tau_min", 0.5), cfg.get("tau_max", 1.5)
    anchor_weight = float(cfg.get("anchor_weight", 0.0))
    init_hard = renderer(mode="hard").detach().clone()

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "t", "sds_loss", "anchor_loss", "colors_used", "lum_levels", "hf_energy", "l1_drift_from_init"])

    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        tau = random.uniform(tau_min, tau_max)
        img = renderer(tau=tau, mode="gumbel")  # (3, H, W)

        if random.random() < cfg.get("hflip_prob", 0.5):
            img = img.flip(-1)

        big = F.interpolate(img.unsqueeze(0), size=guidance.render_size, mode="bilinear")
        sds, t_used = guidance.sds_loss(
            big,
            guidance_scale=float(cfg.get("guidance_scale", 40.0)),
            grad_scale=float(cfg.get("grad_scale", 1.0)),
            t_min=float(cfg.get("t_min", 0.02)),
            t_max=float(cfg.get("t_max", 0.98)),
        )

        anchor = torch.zeros((), device=device)
        if anchor_weight > 0 and reference_small is not None:
            anchor = lowpass_l1_anchor(renderer(tau=1.0, mode="softmax"), reference_small)

        (sds + anchor_weight * anchor).backward()
        if cfg.get("clip_grad", True):
            torch.nn.utils.clip_grad_norm_(renderer.parameters(), cfg.get("max_grad_norm", 1.0))
        opt.step()

        if step % save_steps == 0:
            hard = renderer(mode="hard").detach()
            _save_png(hard, png_dir / f"{step}_hard.png")
            _save_png(renderer(tau=1.0, mode="softmax").detach(), png_dir / f"{step}_soft.png")
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, t_used, f"{float(sds):.4f}", f"{float(anchor):.6f}",
                    colors_used(renderer.hard_indices()),
                    luminance_levels(hard),
                    f"{highfreq_energy(hard):.5f}",
                    f"{l1_to_reference(hard, init_hard):.5f}",
                ])
            print(f"[{step}/{steps}] sds={float(sds):.2f} colors={colors_used(renderer.hard_indices())}")

    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard.png")
    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard_1x.png", resize=None)
    print(f"Done -> {out_dir}")
