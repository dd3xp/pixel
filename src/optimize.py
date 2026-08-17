"""Optimization loops.

P0: single-scale (`run`) — direct optimization at target resolution.
P1: multi-scale pyramid (`run_pyramid`) — refine at each scale, downsample,
re-initialize the next scale from the previous level's result.
"""
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .losses import ANCHORS
from .metrics import colors_used, highfreq_energy, l1_to_reference, luminance_levels
from .palette import load_hex_palette
from .renderer import PaletteRenderer
from .sds import SDXLGuidance


def _load_image(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(img.height, img.width, 3) / 255.0
    return t.permute(2, 0, 1).to(device)  # (3, H, W) full resolution


def _load_mask(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, img.height, img.width) / 255.0
    return t.to(device)  # (1, H, W), 1 = subject, 0 = background


def _resize(image_3hw: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(image_3hw.unsqueeze(0), size=(size, size), mode="bilinear", antialias=True).squeeze(0)


def _dominant_downsample(indices_hw: torch.Tensor, palette: torch.Tensor, size: int) -> torch.Tensor:
    """Mode-pool palette indices per block (pixel-artist style downscale).

    Preserves color identity: thin features keep their color instead of being
    averaged into mud and re-quantized to background. Requires integer ratio.
    """
    H = indices_hw.shape[0]
    f = H // size
    idx = indices_hw[: size * f, : size * f].reshape(size, f, size, f).permute(0, 2, 1, 3).reshape(size, size, f * f)
    mode = idx.mode(dim=-1).values
    return palette[mode].permute(2, 0, 1)


def _save_png(image_3hw: torch.Tensor, path: Path, resize: int | None = 256) -> None:
    arr = (image_3hw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    if resize and resize > img.width:
        img = img.resize((resize, resize), Image.NEAREST)
    img.save(path)


def _build_guidance(cfg: dict, device: str) -> SDXLGuidance:
    guidance = SDXLGuidance(
        model_id=cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
        device=device,
        render_size=int(cfg.get("render_size", 1024)),
    )
    guidance.set_prompt(cfg["prompt"], cfg.get("negative_prompt", ""))
    return guidance


def _optimize_scale(
    renderer: PaletteRenderer,
    guidance: SDXLGuidance,
    cfg: dict,
    out_dir: Path,
    steps: int,
    lr: float,
    anchor_ref: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
    t_max: float | None = None,
    anchor_wmap: torch.Tensor | None = None,  # (1, H, W) per-pixel anchor weight
    guidance_scale: float | None = None,
) -> None:
    """Inner loop shared by single-scale and pyramid runs."""
    device = renderer.logits.device
    png_dir = out_dir / "png_logs"
    png_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.Adam(renderer.parameters(), lr=lr)
    save_steps = int(cfg.get("save_steps", 50))
    tau_min, tau_max = cfg.get("tau_min", 0.5), cfg.get("tau_max", 1.5)
    anchor_fn = ANCHORS[cfg.get("anchor_type", "l1")]
    t_max = float(cfg.get("t_max", 0.98)) if t_max is None else t_max
    init_hard = renderer(mode="hard").detach().clone()

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "t", "sds_loss", "anchor_loss", "colors_used", "lum_levels", "hf_energy", "l1_drift_from_init"])

    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        tau = random.uniform(tau_min, tau_max)
        img = renderer(tau=tau, mode="gumbel")

        if random.random() < cfg.get("hflip_prob", 0.5):
            img = img.flip(-1)

        big = F.interpolate(img.unsqueeze(0), size=guidance.render_size, mode="bilinear")
        sds, t_used = guidance.sds_loss(
            big,
            guidance_scale=float(guidance_scale if guidance_scale is not None else cfg.get("guidance_scale", 40.0)),
            grad_scale=float(cfg.get("grad_scale", 1.0)),
            t_min=float(cfg.get("t_min", 0.02)),
            t_max=t_max,
        )

        anchor = torch.zeros((), device=device)
        if anchor_weight > 0 and anchor_ref is not None:
            if anchor_wmap is not None:
                anchor = (anchor_wmap * (renderer(tau=1.0, mode="softmax") - anchor_ref).abs()).mean()
            else:
                anchor = anchor_fn(renderer(tau=1.0, mode="softmax"), anchor_ref)

        if cfg.get("grad_combine", "sum") == "norm" and anchor_weight > 0 and anchor_ref is not None:
            # Normalize each gradient before combining: anchor_weight becomes a true
            # relative strength instead of being drowned by SDS's raw magnitude.
            sds.backward()
            g_sds = renderer.logits.grad.detach().clone()
            renderer.logits.grad = None
            anchor.backward()
            g_anc = renderer.logits.grad.detach().clone()
            renderer.logits.grad = None
            renderer.logits.grad = g_sds / (g_sds.norm() + 1e-8) + anchor_weight * g_anc / (g_anc.norm() + 1e-8)
        else:
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
            print(f"[{out_dir.name} {step}/{steps}] sds={float(sds):.2f} colors={colors_used(renderer.hard_indices())}", flush=True)

    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard.png")


def run(cfg: dict, out_dir: Path) -> None:
    """P0 single-scale optimization at target resolution."""
    device = cfg.get("device", "cuda")
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    palette = load_hex_palette(cfg["palette"]).to(device)
    size = int(cfg["image_size"])
    renderer = PaletteRenderer(size, size, palette, init_std=cfg.get("init_std", 1.0)).to(device)

    reference_small = None
    if cfg.get("image"):
        reference_small = _resize(_load_image(cfg["image"], device), size)
        renderer.init_from_image(reference_small, distance=cfg.get("init_distance", "l1"))

    guidance = _build_guidance(cfg, device)
    _optimize_scale(
        renderer, guidance, cfg, out_dir, int(cfg["steps"]), float(cfg.get("lr", 0.025)),
        anchor_ref=reference_small, anchor_weight=float(cfg.get("anchor_weight", 0.0)),
    )
    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard_1x.png", resize=None)
    print(f"Done -> {out_dir}", flush=True)


def run_pyramid(cfg: dict, out_dir: Path) -> None:
    """P1 multi-scale: refine at each scale, hand result down to the next."""
    device = cfg.get("device", "cuda")
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    scales: list[int] = [int(s) for s in cfg["scales"]]
    steps_list: list[int] = [int(s) for s in cfg["steps_per_scale"]]
    assert len(scales) == len(steps_list), "scales and steps_per_scale must align"
    n = len(scales)
    lrs = cfg.get("lr_per_scale") or [cfg.get("lr", 0.025)] * n
    aw = cfg.get("anchor_weight", 0.0)
    anchor_weights = aw if isinstance(aw, list) else [float(aw)] * n
    tm = cfg.get("t_max_per_scale")
    t_maxes = [float(x) for x in tm] if tm else [float(cfg.get("t_max", 0.98))] * n
    gs = cfg.get("guidance_scale_per_scale")
    gs_list = [float(x) for x in gs] if gs else [float(cfg.get("guidance_scale", 40.0))] * n
    anchor_mode = cfg.get("anchor_mode", "carry")  # 'carry' = previous level result; 'source' = downsampled source

    palette = load_hex_palette(cfg["palette"]).to(device)
    source = _load_image(cfg["image"], device)
    mask = _load_mask(cfg["mask"], device) if cfg.get("mask") else None
    bg_release_q = cfg.get("bg_release_q")  # None = spatially uniform anchor
    bg_weight_min = float(cfg.get("bg_weight_min", 0.0))
    guidance = _build_guidance(cfg, device)

    prompts = cfg.get("prompt_per_scale") or [cfg["prompt"]] * n
    assert len(prompts) == n, "prompt_per_scale must align with scales"
    carry_mode = cfg.get("carry_mode", "bilinear")  # 'bilinear' (soft render) | 'dominant' (mode-pooled hard indices)
    carry = None  # previous level's soft render, full precision
    carry_indices = None  # previous level's hard palette indices
    renderer = None
    for i, (size, steps) in enumerate(zip(scales, steps_list)):
        level_dir = out_dir / f"level_{i}_{size}"
        reference_small = _resize(source, size)
        if carry is None:
            init_img = reference_small
        elif carry_mode == "dominant":
            init_img = _dominant_downsample(carry_indices, palette, size)
        else:
            init_img = _resize(carry, size)

        renderer = PaletteRenderer(size, size, palette, init_std=cfg.get("init_std", 1.0)).to(device)
        renderer.init_from_image(init_img, distance=cfg.get("init_distance", "l1"))

        if anchor_mode == "mixed" and mask is not None:
            # subject anchors to the color-true source; background to the carry
            m = _resize(mask, size).clamp(0, 1)
            anchor_ref = m * reference_small + (1.0 - m) * init_img
        elif anchor_mode == "flatbg" and mask is not None:
            # subject anchors to source; background to a flat plane of the source's
            # dominant background color (palette-projected) — enforces both correct
            # color and flatness by construction
            m = _resize(mask, size).clamp(0, 1)
            bg_pixels = reference_small[:, m.squeeze(0) < 0.5].T  # (N, 3)
            if bg_pixels.numel() and cfg.get("flatbg_color", "mode") == "mode":
                # most common palette color among bg pixels — robust to striped/mixed bg
                d = (bg_pixels.unsqueeze(1) - palette.unsqueeze(0)).abs().sum(-1)
                flat_color = palette[d.argmin(1).mode().values]
            else:
                med = bg_pixels.T.median(dim=1).values if bg_pixels.numel() else reference_small.mean((1, 2))
                flat_color = palette[(palette - med).abs().sum(1).argmin()]
            flat = flat_color.view(3, 1, 1).expand_as(reference_small)
            anchor_ref = m * reference_small + (1.0 - m) * flat
        elif anchor_mode == "source":
            anchor_ref = reference_small
        else:
            anchor_ref = init_img

        wmap = None
        beta = 1.0
        if mask is not None and bg_release_q is not None:
            S0, T = scales[0], scales[-1]
            frac = (size - T) / (S0 - T) if S0 > T else 0.0
            beta = max(frac, 0.0) ** float(bg_release_q) if size > T else bg_weight_min
            beta = max(beta, bg_weight_min)
            mask_s = _resize(mask, size).clamp(0, 1)
            wmap = mask_s + (1.0 - mask_s) * beta

        if i == 0 or prompts[i] != prompts[i - 1]:
            guidance.set_prompt(prompts[i], cfg.get("negative_prompt", ""))
        print(f"=== level {i}: {size}x{size}, {steps} steps, anchor_w={anchor_weights[i]}, t_max={t_maxes[i]}, bg_beta={beta:.3f} ===", flush=True)
        _optimize_scale(
            renderer, guidance, cfg, level_dir, steps, float(lrs[i]),
            anchor_ref=anchor_ref, anchor_weight=float(anchor_weights[i]), t_max=t_maxes[i],
            anchor_wmap=wmap, guidance_scale=gs_list[i],
        )

        carry = renderer(tau=1.0, mode="softmax").detach()
        carry_indices = renderer.hard_indices()

    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard.png")
    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard_1x.png", resize=None)
    print(f"Done -> {out_dir}", flush=True)
