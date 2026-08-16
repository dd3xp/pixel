"""Score Distillation Sampling against a frozen SDXL teacher.

Teacher choice is isolated behind SDXLGuidance so other teachers (SD1.5,
DeepFloyd, pixel-art LoRA) can be added later with the same interface.
"""
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class SDXLGuidance:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        taesd_id: str = "madebyollin/taesdxl",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        render_size: int = 1024,
    ):
        self.device, self.dtype, self.render_size = device, dtype, render_size

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
        # Tiny VAE: cheap to backprop through; kept in fp32 for gradient stability.
        self.vae = AutoencoderTiny.from_pretrained(taesd_id, torch_dtype=torch.float32).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        for m in (self.text_encoder, self.text_encoder_2, self.unet, self.vae):
            m.requires_grad_(False).eval()

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self._embeds = None

    @torch.no_grad()
    def set_prompt(self, prompt: str, negative_prompt: str = "") -> None:
        """Encode cond+uncond prompt embeddings once (SDXL dual-encoder scheme)."""
        embeds, pooleds = [], []
        for text in (negative_prompt, prompt):
            parts = []
            for tok, enc in ((self.tokenizer, self.text_encoder), (self.tokenizer_2, self.text_encoder_2)):
                ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(self.device)
                out = enc(ids, output_hidden_states=True)
                if enc is self.text_encoder_2:
                    pooleds.append(out.text_embeds)
                parts.append(out.hidden_states[-2])
            embeds.append(torch.cat(parts, dim=-1))
        s = self.render_size
        time_ids = torch.tensor([[s, s, 0, 0, s, s]], device=self.device, dtype=self.dtype)
        self._embeds = {
            "prompt_embeds": torch.cat(embeds).to(self.dtype),          # (2, 77, 2048) uncond first
            "pooled": torch.cat(pooleds).to(self.dtype),                # (2, 1280)
            "time_ids": time_ids.repeat(2, 1),
        }

    def sds_loss(
        self,
        image: torch.Tensor,  # (1, 3, S, S) in [0, 1], requires grad
        guidance_scale: float = 40.0,
        grad_scale: float = 1.0,
        t_min: float = 0.02,
        t_max: float = 0.98,
    ) -> tuple[torch.Tensor, int]:
        assert self._embeds is not None, "call set_prompt() first"
        latents = self.vae.encode(image.float() * 2.0 - 1.0).latents.to(self.dtype)

        t_lo = int(t_min * self.num_train_timesteps)
        t_hi = int(t_max * self.num_train_timesteps)
        t = torch.randint(t_lo, t_hi, (1,), device=self.device)

        noise = torch.randn_like(latents)
        ac_t = self.alphas_cumprod[t].to(self.dtype)
        noisy = ac_t.sqrt() * latents + (1.0 - ac_t).sqrt() * noise

        with torch.no_grad():
            e = self._embeds
            eps = self.unet(
                torch.cat([noisy] * 2),
                t.repeat(2),
                encoder_hidden_states=e["prompt_embeds"],
                added_cond_kwargs={"text_embeds": e["pooled"], "time_ids": e["time_ids"]},
            ).sample
            eps_uncond, eps_text = eps.chunk(2)
            eps_pred = eps_uncond + guidance_scale * (eps_text - eps_uncond)

        w = (1.0 - ac_t)  # 'cumprod' weighting
        grad = (grad_scale * w * (eps_pred - noise)).nan_to_num()
        # SpecifyGradient trick: d(loss)/d(latents) == grad
        loss = 0.5 * F.mse_loss(latents, (latents - grad).detach(), reduction="sum")
        return loss, int(t.item())
