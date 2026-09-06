"""Cycle-1 oracle diagnostics: where is v7's residual -- structure or colour?

Two conditional fine-tunes of v7 (same UNet, same data, 20k steps), each fed a
self-supervised side-condition extracted from the training sprite itself:

  --cond struct : S = [alpha, 4-level OKLab-lightness "value sketch"] (2ch),
                  concatenated to the noisy RGBA as extra input channels
                  (conv_in 4->6, new channels zero-init so step 0 == v7).
  --cond paltok : global palette = 8 most frequent opaque colours sorted by
                  luminance -> 8 learned tokens appended to the CLIP text tokens
                  (cross-attention). Colours stay continuous; nothing is snapped.

Both use CDM-style conditioning augmentation (Gaussian noise on the condition)
and 10% condition dropout so the model keeps an unconditional fallback.
Evaluation (sample_cond.py) feeds ORACLE conditions from held-out real sprites
disjoint from the FD reference set; the resulting FD is an upper bound on what
a learned stage-1 could unlock. Oracle FD << 64.8 => that modality is v7's
bottleneck and worth a stage-1; oracle FD ~ 64.8 => it is not.
"""
import argparse
import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

BUCKETS = [12, 16, 20, 24, 32, 48, 64]
LOW = [0, 1, 2, 3]
BATCH = {12: 320, 16: 256, 20: 224, 24: 192, 32: 128, 48: 64, 64: 40}
EVAL_SIZES = [12, 16, 20, 24]
K_PAL = 8
N_LEVELS = 4


# ----------------------------------------------------------------- data (== v7)
def downscale_rgba(im, side):
    f = side / max(im.size)
    w, h = max(1, round(im.width * f)), max(1, round(im.height * f))
    a = np.array(im).astype(np.float32)
    a[:, :, :3] *= a[:, :, 3:4] / 255.0
    pm = Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA").resize((w, h), Image.BOX)
    b = np.array(pm).astype(np.float32)
    alpha = b[:, :, 3:4]
    b[:, :, :3] = np.where(alpha > 0, b[:, :, :3] / np.maximum(alpha, 1) * 255.0, 0)
    return Image.fromarray(b.clip(0, 255).astype(np.uint8), "RGBA")


def to_tensor(im, side):
    if max(im.size) > side:
        im = downscale_rgba(im, side)
    elif max(im.size) * 2 <= side:
        f = side // max(im.size)
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    a = np.array(canvas).astype(np.float32)
    a[a[:, :, 3] < 128] = 0.0
    a[:, :, 3] = (a[:, :, 3] >= 128) * 255.0
    return torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1.0


class NativeSprites(torch.utils.data.Dataset):
    def __init__(self, sources):
        self.rows, self.bucket_of = [], []
        for img_dir, captions_csv, repeat in sources:
            img_dir = Path(img_dir)
            with open(captions_csv, newline="", encoding="utf-8") as f:
                rows = [(img_dir / r["path"], r["text"]) for r in csv.DictReader(f)]
            for path, text in rows:
                s = max(Image.open(path).size)
                b = next((i for i, bb in enumerate(BUCKETS) if bb >= s), len(BUCKETS) - 1)
                self.rows.extend([(path, text)] * repeat)
                self.bucket_of.extend([b] * repeat)
                for t in LOW:
                    if t != b and s >= BUCKETS[t] * 1.25:
                        self.rows.extend([(path, text)] * repeat)
                        self.bucket_of.extend([t] * repeat)
            print(f"source {img_dir}: {len(rows)} x{repeat}", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        b = self.bucket_of[i]
        return to_tensor(Image.open(path).convert("RGBA"), BUCKETS[b]), text, b


class BucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket_of, steps):
        self.groups = {b: [i for i, x in enumerate(bucket_of) if x == b] for b in range(len(BUCKETS))}
        self.groups = {b: g for b, g in self.groups.items() if len(g) >= 8}
        self.keys = sorted(self.groups)
        self.weights = [len(self.groups[b]) for b in self.keys]
        self.steps = steps

    def __iter__(self):
        for _ in range(self.steps):
            b = random.choices(self.keys, weights=self.weights)[0]
            yield random.sample(self.groups[b], min(BATCH[BUCKETS[b]], len(self.groups[b])))

    def __len__(self):
        return self.steps


# ------------------------------------------------------- self-supervised conditions
def oklab_L(rgb01):
    """rgb01 (B,3,H,W) in [0,1] -> OKLab lightness (B,H,W) in ~[0,1]."""
    lin = torch.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l.clamp_min(0) ** (1 / 3), m.clamp_min(0) ** (1 / 3), s.clamp_min(0) ** (1 / 3)
    return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s


@torch.no_grad()
def make_struct(x):
    """x (B,4,H,W) in [-1,1] -> S (B,2,H,W): [alpha(+-1), 4-level value sketch]."""
    alpha = x[:, 3:4]
    L = oklab_L((x[:, :3] + 1) / 2)
    op = alpha[:, 0] > 0
    Lq = torch.zeros_like(L)
    for i in range(L.shape[0]):
        if op[i].any():
            v = L[i][op[i]]
            q = ((L[i] - v.min()) / (v.max() - v.min() + 1e-6) * N_LEVELS).floor().clamp(0, N_LEVELS - 1)
            Lq[i] = torch.where(op[i], q / (N_LEVELS - 1) * 2 - 1, torch.zeros_like(q))
    return torch.cat([alpha, Lq.unsqueeze(1)], 1)


@torch.no_grad()
def make_palette(x, K=K_PAL):
    """x (B,4,H,W) -> (B,K,3) in [-1,1]: K most frequent opaque colours, sorted by
    luminance, padded by repeating the brightest. Exact (pixel art has few colours)."""
    B = x.shape[0]
    out = torch.zeros(B, K, 3, device=x.device)
    lumw = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device)
    for i in range(B):
        op = x[i, 3] > 0
        if not op.any():
            continue
        rgb = x[i, :3][:, op].t()
        codes = ((rgb + 1) * 127.5).round().clamp(0, 255).long()
        key = codes[:, 0] * 65536 + codes[:, 1] * 256 + codes[:, 2]
        _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
        order = cnt.argsort(descending=True)[:K]
        cols = torch.stack([rgb[inv == j][0] for j in order])
        cols = cols[(((cols + 1) / 2) @ lumw).argsort()]
        k = cols.shape[0]
        out[i, :k] = cols
        if k < K:
            out[i, k:] = cols[-1]
    return out


class PalTok(nn.Module):
    """8 palette colours -> 8 cross-attention tokens (dim 512), with a learned
    null token set used when the palette condition is dropped."""

    def __init__(self, K=K_PAL, D=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(3, D), nn.SiLU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(K, D))
        self.null = nn.Parameter(torch.zeros(1, K, D))

    def forward(self, pal, drop=None):
        tok = self.proj(pal) + self.pos
        if drop is not None:
            tok = torch.where(drop.view(-1, 1, 1), self.null.expand_as(tok), tok)
        return tok


# ------------------------------------------------------------------ model utils
def build_unet(in_ch):
    return UNet2DConditionModel(
        sample_size=64, in_channels=in_ch, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    )


def load_v7_into(model, ckpt, in_ch, device):
    sd = torch.load(ckpt, map_location=device)
    if "unet" in sd:  # our own checkpoint format
        sd = sd["unet"]
    w = sd["conv_in.weight"]
    if w.shape[1] != in_ch:  # zero-init the new condition channels: step 0 == v7
        new = torch.zeros(w.shape[0], in_ch, *w.shape[2:], dtype=w.dtype, device=w.device)
        new[:, : w.shape[1]] = w
        sd["conv_in.weight"] = new
    model.load_state_dict(sd)


@torch.no_grad()
def embed(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


def make_grid(images, cols=8, scale=None):
    n, _, h, w = images.shape
    scale = scale or max(1, 256 // h)
    rows = math.ceil(n / cols)
    grid = torch.ones(4, rows * h, cols * w)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i]
    rgb, alpha = grid[:3], grid[3:4].clamp(0, 1)
    yy, xx = torch.meshgrid(torch.arange(rows * h), torch.arange(cols * w), indexing="ij")
    cell = max(1, h // 4)
    checker = (((yy // cell + xx // cell) % 2) * 0.12 + 0.82).unsqueeze(0)
    out = rgb * alpha + checker * (1 - alpha)
    img = Image.fromarray((out.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy())
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size, S=None, device="cuda", steps=100, cfg=4.0):
    """CFG on text only; the side-condition (S channels / palette tokens already
    inside cond & uncond) is kept in both branches."""
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    x = torch.randn(n, 4, size, size, device=device)
    for t in scheduler.timesteps:
        inp = torch.cat([x, S], 1) if S is not None else x
        e_c = model(inp, t, encoder_hidden_states=cond, class_labels=lab).sample
        e_u = model(inp, t, encoder_hidden_states=uncond, class_labels=lab).sample
        x = scheduler.step(e_u + cfg * (e_c - e_u), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True, choices=["struct", "paltok"])
    p.add_argument("--init", required=True, help="v7 checkpoint")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", required=True)
    p.add_argument("--sample_every", type=int, default=4000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cond_drop", type=float, default=0.1)
    p.add_argument("--cond_noise", type=float, default=0.3, help="max sigma of condition augmentation")
    args = p.parse_args()
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    sources = [
        ("data/oga_clean", "data/oga_captions.csv", 1),
        ("data/extra_all", "data/extra_all.csv", 1),
        ("data/oga_clean", "data/tool_candidates.csv", 2),
    ]
    ds = NativeSprites(sources)
    print(f"dataset: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    in_ch = 6 if args.cond == "struct" else 4
    model = build_unet(in_ch).to(device)
    load_v7_into(model, args.init, in_ch, device)
    print(f"unet params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M, init from {args.init}", flush=True)
    paltok = PalTok().to(device) if args.cond == "paltok" else None
    params = list(model.parameters()) + (list(paltok.parameters()) if paltok else [])

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(params, lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema > 0 else None
    ema_pal = copy.deepcopy(paltok).eval().requires_grad_(False) if (paltok and args.ema > 0) else None

    # fixed real references for the in-training sample grids (oracle conditions)
    ref_rows = [ds.rows[i] for i in random.sample(range(len(ds)), 8)]
    ref_cond = embed([t for _, t in ref_rows], tokenizer, text_encoder, device)
    ref_uncond = embed([""] * 8, tokenizer, text_encoder, device)

    def side_condition(x, train):
        """Returns (S or None, extra_tokens or None) for a batch x, with augmentation if train."""
        B = x.shape[0]
        drop = (torch.rand(B, device=device) < args.cond_drop) if train else torch.zeros(B, dtype=torch.bool, device=device)
        if args.cond == "struct":
            S = make_struct(x)
            if train:
                sig = torch.rand(B, 1, 1, 1, device=device) * args.cond_noise * (torch.rand(B, 1, 1, 1, device=device) < 0.5)
                S = S + sig * torch.randn_like(S)
                S = torch.where(drop.view(-1, 1, 1, 1), torch.zeros_like(S), S)
            return S, None
        pal = make_palette(x)
        if train:
            sig = torch.rand(B, 1, 1, device=device) * (args.cond_noise / 3) * (torch.rand(B, 1, 1, device=device) < 0.5)
            pal = (pal + sig * torch.randn_like(pal)).clamp(-1, 1)
        net_pal = paltok if train else (ema_pal or paltok)
        return None, net_pal(pal, drop)

    step = 0
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)
        x, b = x.to(device), b.to(device)
        S, extra = side_condition(x, train=True)
        if extra is not None:
            cond = torch.cat([cond, extra], 1)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (x.shape[0],), device=device)
        xt = scheduler.add_noise(x, noise, t)
        inp = torch.cat([xt, S], 1) if S is not None else xt
        pred = model(inp, t, encoder_hidden_states=cond, class_labels=b).sample
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)
                if ema_pal is not None:
                    for pe, pm in zip(ema_pal.parameters(), paltok.parameters()):
                        pe.lerp_(pm, 1.0 - args.ema)
        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} bucket={BUCKETS[int(b[0])]}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                xr = torch.stack([to_tensor(Image.open(pth).convert("RGBA"), s) for pth, _ in ref_rows]).to(device)
                Sr, extra_r = side_condition(xr, train=False)
                c, u = ref_cond, ref_uncond
                if extra_r is not None:
                    c, u = torch.cat([c, extra_r], 1), torch.cat([u, extra_r], 1)
                torch.manual_seed(args.seed)
                gen = sample(net, scheduler, c, u, s, S=Sr)
                both = torch.cat([((xr + 1) / 2).clamp(0, 1).cpu(), gen], 0)  # row 1 = real ref, row 2 = generated
                make_grid(both).save(out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save({"cond": args.cond, "unet": net.state_dict(),
                        "paltok": (ema_pal or paltok).state_dict() if paltok else None}, out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
