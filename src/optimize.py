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

from .losses import ANCHORS, _gaussian_blur
from .metrics import colors_used, highfreq_energy, l1_to_reference, luminance_levels
from .palette import load_hex_palette
from .renderer import PaletteRenderer
from .sds import SDXLGuidance


def _load_image(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(img.height, img.width, 3) / 255.0
    return t.permute(2, 0, 1).to(device)  # (3, H, W) full resolution


def _tone_boost(img: torch.Tensor, mask: torch.Tensor | None, gain: float = 2.0,
                unsharp: float = 0.5, sigma: float = 4.0) -> torch.Tensor:
    """Exaggerate subject tonal contrast so shading survives palette quantization
    (what a pixel artist does by hand: push shadows down a ramp, highlights up).
    """
    w = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(3, 1, 1)
    lum = (img * w).sum(0, keepdim=True)
    m = mask if mask is not None else torch.ones_like(lum)
    mean = (lum * m).sum() / m.sum().clamp(min=1e-6)
    delta = lum - mean
    gain_lo = gain if isinstance(gain, float) else gain  # symmetric fallback
    if isinstance(gain, (tuple, list)):
        gain_hi, gain_lo = float(gain[0]), float(gain[1])
    else:
        gain_hi = gain_lo = float(gain)
    boosted = torch.where(delta > 0, gain_hi * delta, gain_lo * delta) + mean
    scale = (boosted / lum.clamp(min=1e-4)).clamp(0.2, 3.0)
    out = img * scale
    out = out + unsharp * (out - _gaussian_blur(out, sigma))
    out = out.clamp(0, 1)
    return m * out + (1 - m) * img


def _cluster_reference(ref_3hw: torch.Tensor, mask_1hw: torch.Tensor, palette: torch.Tensor,
                       k: int = 4, sigma: float = 1.0, iters: int = 8) -> torch.Tensor:
    """Hue-aware coherent reference: blur, k-means the subject pixels in RGB
    (so the blue center keeps its own cluster), snap each cluster to its nearest
    palette color. Yields clean shading regions without destroying hue."""
    img = _gaussian_blur(ref_3hw, sigma)
    subject = mask_1hw.reshape(img.shape[1], img.shape[2]) > 0.5
    out = ref_3hw.clone()
    px = img[:, subject].T  # (N, 3)
    if px.shape[0] < k:
        return out
    w = torch.tensor([0.299, 0.587, 0.114], device=px.device)
    order = (px @ w).argsort()
    seeds = px[order[torch.linspace(0, px.shape[0] - 1, k, device=px.device).long()]]
    for _ in range(iters):
        assign = (px.unsqueeze(1) - seeds.unsqueeze(0)).abs().sum(-1).argmin(1)
        for c in range(k):
            sel = assign == c
            if sel.any():
                seeds[c] = px[sel].mean(0)
    amap = torch.zeros(img.shape[1], img.shape[2], dtype=torch.long, device=img.device)
    amap[subject] = assign
    for c in range(k):
        sel = subject & (amap == c)
        if sel.any():
            color = palette[(palette - seeds[c]).abs().sum(1).argmin()]
            out[:, sel] = color.view(3, 1)
    return out


def _posterize_shading(ref_3hw: torch.Tensor, mask_1hw: torch.Tensor, palette: torch.Tensor,
                       bands: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Rebuild the subject as coherent tone bands (pixel-art ramp shading):
    blur luminance -> quantile-split into bands -> each band gets the palette
    color nearest its mean color. Turns mushy gradients into clean shading regions.
    """
    w = torch.tensor([0.299, 0.587, 0.114], device=ref_3hw.device).view(3, 1, 1)
    lum = _gaussian_blur((ref_3hw * w).sum(0, keepdim=True).repeat(3, 1, 1), sigma)[0]
    subject = mask_1hw.squeeze(0) > 0.5
    out = ref_3hw.clone()
    ls = lum[subject]
    if ls.numel() < bands:
        return out
    qs = torch.quantile(ls, torch.linspace(0, 1, bands + 1, device=ls.device)[1:-1])
    band_of = torch.bucketize(lum, qs)  # (H, W) in [0, bands-1]
    for b in range(bands):
        sel = subject & (band_of == b)
        if sel.any():
            mean_color = ref_3hw[:, sel].mean(1)
            color = palette[(palette - mean_color).abs().sum(1).argmin()]
            out[:, sel] = color.view(3, 1)
    return out


def _valley_expand(img: torch.Tensor, mask: torch.Tensor | None, kernel: int = 33,
                   gain: float = 1.5, sigma: float = 8.0) -> torch.Tensor:
    """PixelOE-style outline expansion: thin dark valleys (inter-petal shadows,
    contour lines) are detected and thickened so they survive extreme downscaling
    as 1px separators — the pixel-artist's 'carve the gaps' move."""
    w = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(3, 1, 1)
    lum = (img * w).sum(0, keepdim=True)
    local_mean = _gaussian_blur(lum.repeat(3, 1, 1), sigma)[:1]
    valley = (local_mean - lum).clamp(min=0)  # positive where darker than surroundings
    # gate: only thin valleys BETWEEN bright regions (petal gaps), not dark areas themselves
    valley = valley * (local_mean > 0.55).float()
    pad = kernel // 2
    valley = F.max_pool2d(valley.unsqueeze(0), kernel, stride=1, padding=pad).squeeze(0)
    out = (img - gain * valley).clamp(0, 1)
    if mask is not None:
        out = mask * out + (1 - mask) * img
    return out


def _gap_tophat(img: torch.Tensor, mask: torch.Tensor | None, kernel: int = 41,
                gain: float = 1.2, cap: float = 0.8,
                target_color: torch.Tensor | None = None) -> torch.Tensor:
    """Thin-structure-selective gap carving via morphological black top-hat:
    closing(lum) - lum is large ONLY for structures narrower than the kernel
    (petal gaps), ~zero inside wide dark regions (flower center). Gap pixels are
    blended TOWARD the background color (the artist's separator), not to black."""
    w = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(3, 1, 1)
    lum = (img * w).sum(0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
    pad = kernel // 2
    dilated = F.max_pool2d(lum, kernel, stride=1, padding=pad)
    closing = -F.max_pool2d(-dilated, kernel, stride=1, padding=pad)
    tophat = (closing - lum).clamp(min=0).squeeze(0)  # thin dark lines only
    a = (gain * tophat).clamp(0, cap)  # blend strength, capped
    if target_color is None:
        target = torch.zeros_like(img)
    else:
        target = target_color.view(3, 1, 1).expand_as(img)
    out = (img * (1 - a) + target * a).clamp(0, 1)
    if mask is not None:
        out = mask * out + (1 - mask) * img
    return out


def _load_mask(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, img.height, img.width) / 255.0
    return t.to(device)  # (1, H, W), 1 = subject, 0 = background


def _resize(image_3hw: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    if mode == "lanczos":
        arr = (image_3hw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
        t = torch.tensor(list(img.getdata()), dtype=torch.float32).view(size, size, 3) / 255.0
        return t.permute(2, 0, 1).to(image_3hw.device)
    if mode == "kcentroid":
        return _kcentroid_downsample(image_3hw, size)
    if mode == "kcentroid_dither":
        return _kcentroid_downsample(image_3hw, size, dither=True)
    return F.interpolate(image_3hw.unsqueeze(0), size=(size, size), mode="bilinear", antialias=True).squeeze(0)


def _kcentroid_downsample(image_3hw: torch.Tensor, size: int, factor: int = 4, iters: int = 4,
                          dither: bool = False) -> torch.Tensor:
    """Pixel-artist style content-aware downscale: per output cell, 2-means
    cluster the cell's pixels and take the dominant cluster's centroid — thin
    features keep their color instead of being averaged into the background.
    """
    inter = F.interpolate(image_3hw.unsqueeze(0), size=(size * factor, size * factor),
                          mode="bilinear", antialias=True).squeeze(0)
    cells = inter.reshape(3, size, factor, size, factor).permute(1, 3, 0, 2, 4).reshape(size, size, 3, factor * factor)
    cells = cells.permute(0, 1, 3, 2)  # (size, size, N, 3)

    lum = cells @ torch.tensor([0.299, 0.587, 0.114], device=cells.device)
    c0 = torch.gather(cells, 2, lum.argmin(-1)[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    c1 = torch.gather(cells, 2, lum.argmax(-1)[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    for _ in range(iters):
        d0 = (cells - c0.unsqueeze(2)).abs().sum(-1)
        d1 = (cells - c1.unsqueeze(2)).abs().sum(-1)
        assign = (d1 < d0).float().unsqueeze(-1)  # 1 -> cluster1
        w1 = assign.sum(2).clamp(min=1e-6)
        w0 = (1 - assign).sum(2).clamp(min=1e-6)
        c1 = (cells * assign).sum(2) / w1
        c0 = (cells * (1 - assign)).sum(2) / w0
    frac1 = assign.mean(2).squeeze(-1)  # weight of cluster 1 per cell
    dominant = (frac1 >= 0.5).unsqueeze(-1)
    out = torch.where(dominant, c1, c0)  # (size, size, 3)
    if dither:
        # cells straddling a gradient encode it as a checkerboard of the two tones
        # (pixel-art dithering) instead of collapsing to one color
        w = torch.tensor([0.299, 0.587, 0.114], device=out.device)
        dark_first = (c0 @ w <= c1 @ w).unsqueeze(-1)
        c_dark = torch.where(dark_first, c0, c1)
        c_light = torch.where(dark_first, c1, c0)
        color_dist = (c0 - c1).abs().sum(-1)
        mixed = (frac1 > 0.35) & (frac1 < 0.65) & (color_dist > 0.12)
        yy, xx = torch.meshgrid(torch.arange(size, device=out.device),
                                torch.arange(size, device=out.device), indexing="ij")
        pick_dark = ((yy + xx) % 2 == 0).unsqueeze(-1)
        dithered = torch.where(pick_dark, c_dark, c_light)
        out = torch.where(mixed.unsqueeze(-1), dithered, out)
    return out.permute(2, 0, 1)


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
        controlnet_id=cfg.get("controlnet_id"),
        lora_id=cfg.get("lora_id"),
        lora_weight_name=cfg.get("lora_weight_name"),
        lora_scale=float(cfg.get("lora_scale", 1.0)),
    )
    guidance.set_prompt(cfg["prompt"], cfg.get("negative_prompt", ""))
    return guidance


def _canny_control(source_3hw: torch.Tensor, size: int) -> torch.Tensor:
    import cv2
    import numpy as np
    arr = (source_3hw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    arr = cv2.resize(arr, (size, size))
    edges = cv2.Canny(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 100, 200)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    t = torch.from_numpy(np.asarray(edges)).float() / 255.0
    return t.unsqueeze(0).repeat(3, 1, 1).to(source_3hw.device)


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

        big = F.interpolate(img.unsqueeze(0), size=guidance.render_size,
                            mode=cfg.get("upscale_mode", "bilinear"))
        sds, t_used = guidance.sds_loss(
            big,
            guidance_scale=float(guidance_scale if guidance_scale is not None else cfg.get("guidance_scale", 40.0)),
            grad_scale=float(cfg.get("grad_scale", 1.0)),
            t_min=float(cfg.get("t_min", 0.02)),
            t_max=t_max,
            controlnet_scale=float(cfg.get("controlnet_scale", 0.0)),
        )

        anchor = torch.zeros((), device=device)
        if anchor_weight > 0 and anchor_ref is not None:
            if anchor_wmap is not None:
                anchor = (anchor_wmap * (renderer(tau=1.0, mode="softmax") - anchor_ref).abs()).mean()
            else:
                anchor = anchor_fn(renderer(tau=1.0, mode="softmax"), anchor_ref)
            cw = float(cfg.get("coherence_weight", 0.0))
            if cw > 0:
                # spatial coherence: neighboring pixels should agree on palette choice
                p = F.softmax(renderer.logits, dim=-1)
                tv = (p[1:, :] - p[:-1, :]).abs().mean() + (p[:, 1:] - p[:, :-1]).abs().mean()
                anchor = anchor + cw * tv

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
    me = int(cfg.get("mask_erode", 0))
    if mask is not None and me > 0:
        # shrink subject region: boundary ring becomes background (flat-anchored,
        # snapped) — keeps bg texture fragments out of the subject reference
        mask = -F.max_pool2d(-mask.unsqueeze(0), me * 2 + 1, stride=1, padding=me).squeeze(0)
    if cfg.get("tone_gain"):
        tg = cfg["tone_gain"]
        source = _tone_boost(source, mask, gain=tg if isinstance(tg, list) else float(tg),
                             unsharp=float(cfg.get("tone_unsharp", 0.5)))
    if cfg.get("gap_gain"):
        bg_col = None
        gmask = mask
        if mask is not None:
            bg_px = source[:, mask.squeeze(0) < 0.5].T
            if bg_px.shape[0] > 50_000:
                bg_px = bg_px[torch.randperm(bg_px.shape[0], device=bg_px.device)[:50_000]]
            bg_col = bg_px.median(dim=0).values
            gme = int(cfg.get("gap_mask_erode", 0))
            if gme > 0:
                # carve only deep inside the subject; keeps boundary-adjacent bg
                # texture (e.g. stripes) out of the top-hat's reach
                gmask = -F.max_pool2d(-mask.unsqueeze(0), gme * 2 + 1, stride=1, padding=gme).squeeze(0)
        source = _gap_tophat(source, gmask, kernel=int(cfg.get("gap_kernel", 41)),
                             gain=float(cfg["gap_gain"]), cap=float(cfg.get("gap_cap", 0.8)),
                             target_color=bg_col)
    elif cfg.get("valley_gain"):
        source = _valley_expand(source, mask, kernel=int(cfg.get("valley_kernel", 33)),
                                gain=float(cfg["valley_gain"]))
    bg_release_q = cfg.get("bg_release_q")  # None = spatially uniform anchor
    bg_weight_min = float(cfg.get("bg_weight_min", 0.0))
    guidance = _build_guidance(cfg, device)
    if cfg.get("controlnet_scale", 0) and cfg.get("controlnet_id"):
        guidance.set_control_image(_canny_control(source, guidance.render_size))

    prompts = cfg.get("prompt_per_scale") or [cfg["prompt"]] * n
    assert len(prompts) == n, "prompt_per_scale must align with scales"
    carry_mode = cfg.get("carry_mode", "bilinear")  # 'bilinear' (soft render) | 'dominant' (mode-pooled hard indices)

    # Compute the flat background color ONCE at full resolution, where the bg
    # content is still distinct (per-level recomputation degrades to the color
    # of downsampling mush).
    flat_color = None
    if mask is not None and anchor_mode == "flatbg":
        bg_px = source[:, mask.squeeze(0) < 0.5].T  # (N, 3)
        if bg_px.shape[0] > 100_000:
            bg_px = bg_px[torch.randperm(bg_px.shape[0], device=bg_px.device)[:100_000]]
        if cfg.get("flatbg_color", "mode") == "mode":
            d = (bg_px.unsqueeze(1) - palette.unsqueeze(0)).abs().sum(-1)
            flat_color = palette[d.argmin(1).mode().values]
        else:
            med = bg_px.median(dim=0).values
            flat_color = palette[(palette - med).abs().sum(1).argmin()]
        print(f"flat bg color: {[round(float(c), 3) for c in flat_color]}", flush=True)
    carry = None  # previous level's soft render, full precision
    carry_indices = None  # previous level's hard palette indices
    renderer = None
    ds_mode = cfg.get("downsample", "bilinear")  # bilinear | lanczos | kcentroid
    shading_bands = cfg.get("shading_bands")
    for i, (size, steps) in enumerate(zip(scales, steps_list)):
        level_dir = out_dir / f"level_{i}_{size}"
        reference_small = _resize(source, size, mode=ds_mode)
        if shading_bands and mask is not None:
            m_s = _resize(mask, size).reshape(1, size, size)
            reference_small = _posterize_shading(reference_small, m_s, palette, bands=int(shading_bands))
        if cfg.get("cluster_ref_k") and mask is not None:
            m_s = _resize(mask, size).reshape(1, size, size)
            reference_small = _cluster_reference(reference_small, m_s, palette, k=int(cfg["cluster_ref_k"]))
        if carry is None:
            init_img = reference_small
        elif carry_mode == "dominant":
            init_img = _dominant_downsample(carry_indices, palette, size)
        else:
            init_img = _resize(carry, size, mode=ds_mode)

        renderer = PaletteRenderer(size, size, palette, init_std=cfg.get("init_std", 1.0)).to(device)
        renderer.init_from_image(init_img, distance=cfg.get("init_distance", "l1"),
                                 sharpness=float(cfg.get("init_sharpness", 10.0)))

        if anchor_mode == "mixed" and mask is not None:
            # subject anchors to the color-true source; background to the carry
            m = _resize(mask, size).clamp(0, 1)
            anchor_ref = m * reference_small + (1.0 - m) * init_img
        elif anchor_mode == "flatbg" and mask is not None:
            # subject anchors to source; background to a flat plane of the source's
            # dominant background color (computed once at full res) — enforces both
            # correct color and flatness by construction
            m = _resize(mask, size).clamp(0, 1)
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

    if cfg.get("bg_snap") and mask is not None and flat_color is not None:
        # final compositing pass: snap background-region pixels to the flat bg color
        # (the pixel-artist "lay down the background" step)
        m_final = _resize(mask, scales[-1]).reshape(scales[-1], scales[-1])
        flat_idx = (palette - flat_color).abs().sum(1).argmin()
        with torch.no_grad():
            bg_sel = m_final < float(cfg.get("bg_snap_threshold", 0.3))
            sel = renderer.logits.data[bg_sel]  # (N, K)
            sel.fill_(-10.0)
            sel[:, int(flat_idx)] = 10.0
            renderer.logits.data[bg_sel] = sel

    if cfg.get("despeckle"):
        # value-aware cleanup: an isolated pixel is removed only when its color is
        # FAR from every neighbor (a true speckle); ramp-adjacent accents (shading,
        # highlights near similar tones) are kept
        far_thresh = float(cfg.get("despeckle_thresh", 0.35))
        with torch.no_grad():
            idx = renderer.hard_indices()
            H, W = idx.shape
            for y in range(H):
                for x in range(W):
                    neigh = [idx[yy, xx] for yy in range(max(0, y - 1), min(H, y + 2))
                             for xx in range(max(0, x - 1), min(W, x + 2)) if (yy, xx) != (y, x)]
                    neigh_t = torch.stack(neigh)
                    if (neigh_t != idx[y, x]).all():
                        own = palette[idx[y, x]]
                        d_min = (palette[neigh_t] - own).abs().sum(-1).min()
                        if d_min > far_thresh:
                            new_idx = int(neigh_t.mode().values)
                            renderer.logits.data[y, x, :] = -10.0
                            renderer.logits.data[y, x, new_idx] = 10.0

    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard.png")
    _save_png(renderer(mode="hard").detach(), out_dir / "final_hard_1x.png", resize=None)
    print(f"Done -> {out_dir}", flush=True)
