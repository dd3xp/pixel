"""Decoupled image cross-attention (IP-Adapter style) for our UNet.

Why not prefix concatenation
----------------------------
v8 appended the image tokens to the text tokens and let one attention handle
both.  Measured outcome: the image branch carried real but weak information
(GAP 0.0105 vs a 0.0017 floor), and with informative text it was *worse* than
the text-only model (GAP 0.030 vs 0.0388) -- the image tokens were competing
with the text for the same attention mass.

Here each cross-attention layer gets its OWN K/V for the image tokens.  Text
attention is computed exactly as before and the image attention is added:

    out = Attn(q, k_text, v_text) + scale * Attn(q, k_img, v_img)

`to_v_ip` is zero-initialised, so at step 0 the model is bit-for-bit the
text-only model; the image path can only ever add, never subtract, the text
signal.  `scale` starts at 1, NOT 0 -- see the note in __init__: zeroing both
deadlocks the branch (each one's gradient is proportional to the other).

Usage:
    procs = install_ip_attn(unet, image_dim=512)   # returns the new params
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IPAttnProcessor(nn.Module):
    """Text tokens are the first `text_len` of encoder_hidden_states; the rest
    are image tokens routed through the extra K/V."""

    def __init__(self, hidden_size, cross_dim, text_len=77):
        super().__init__()
        self.text_len = text_len
        self.to_k_ip = nn.Linear(cross_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_dim, hidden_size, bias=False)
        nn.init.normal_(self.to_k_ip.weight, std=0.02)
        nn.init.zeros_(self.to_v_ip.weight)          # image path starts silent
        # scale must NOT also start at zero: d(loss)/d(scale) is proportional to
        # the image attention output (zero while to_v_ip is zero) and
        # d(loss)/d(to_v_ip) is proportional to scale -- zero-initialising both
        # deadlocks the branch permanently (observed in the first v9 run:
        # |scale| stayed 0.0000).  Start it at 1 and let to_v_ip alone provide
        # the silent start.
        self.scale = nn.Parameter(torch.ones(1))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        ctx = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        img_ctx = None
        if encoder_hidden_states is not None and ctx.shape[1] > self.text_len:
            img_ctx = ctx[:, self.text_len:]
            ctx = ctx[:, :self.text_len]
        if attn.norm_cross is not None and encoder_hidden_states is not None:
            ctx = attn.norm_encoder_hidden_states(ctx)

        batch = hidden_states.shape[0]
        q = attn.to_q(hidden_states)
        k = attn.to_k(ctx)
        v = attn.to_v(ctx)
        head_dim = q.shape[-1] // attn.heads

        def heads(t):
            return t.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v),
                                             attn_mask=attention_mask, dropout_p=0.0)
        if img_ctx is not None:
            ik = self.to_k_ip(img_ctx)
            iv = self.to_v_ip(img_ctx)
            ip = F.scaled_dot_product_attention(heads(q), heads(ik), heads(iv),
                                                attn_mask=None, dropout_p=0.0)
            out = out + self.scale * ip

        out = out.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def install_ip_attn(unet, cross_dim=512, text_len=77):
    """Replace every CROSS-attention processor (attn2) with the decoupled one.
    Self-attention (attn1) is left untouched.  Returns the new module list so
    the caller can put its parameters in the optimiser and save them."""
    procs = {}
    for name in unet.attn_processors.keys():
        if not name.endswith("attn2.processor"):
            procs[name] = unet.attn_processors[name]
            continue
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            i = int(name[len("up_blocks.")])
            hidden = list(reversed(unet.config.block_out_channels))[i]
        elif name.startswith("down_blocks"):
            i = int(name[len("down_blocks.")])
            hidden = unet.config.block_out_channels[i]
        else:
            raise ValueError(name)
        # new modules are born on CPU; follow the UNet's device/dtype
        procs[name] = IPAttnProcessor(hidden, cross_dim, text_len).to(
            device=unet.device, dtype=unet.dtype)
    unet.set_attn_processor(procs)
    return nn.ModuleList([p for p in unet.attn_processors.values()
                          if isinstance(p, IPAttnProcessor)])
