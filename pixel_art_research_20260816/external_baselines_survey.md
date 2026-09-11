# External baselines survey: text-to-pixel-art for very-low-resolution RGBA sprites

Compiled 2026-09-12. Scope: methods from 2020–2026 (plus older classics that are still the standard pixelization baselines) that generate pixel art from text, or that could be chained after a text-to-image model, judged against our target: **12 / 16 / 20 / 24 px square RGBA sprites on a transparent background**.

The paper's current external baselines are gpt-image-2 and SDXL, both generated at 1024 px and downscaled. A reviewer can dismiss these as "not designed for pixel art". This survey lists the pixel-art-specific competitors and says which ones we can actually run.

## Verification legend

- **[V]**: I fetched the link during this survey and the page confirmed the claim.
- **[V-search]**: the link came up in a live web search with a matching title and snippet, but I did not fetch the page (it was blocked, or the fetch returned nothing useful).
- **[U]**: could not verify. The claim comes from search snippets or third-party pages only. Check it before citing.
- A dash means the source did not say. Runtime numbers labelled "estimate" are my own guesses, not measurements.

---

## 1. Text-to-pixel-art generative models (diffusion / GAN / fine-tunes)

### 1a. Academic papers

| Method | Authors, venue, year | Link | Code / weights | Native 16–24 px? | Alpha | Cost | How hard to run |
|---|---|---|---|---|---|---|---|
| **PixDiff-PIG: Palette-Informed Diffusion for Pixel Art Generation** | Authors and venue **[U]** (ResearchGate returned 403) | ResearchGate pub. 398825200 **[V-search]** https://www.researchgate.net/publication/398825200 | No code found | Unclear. It is a TinyUNet DDPM on "low-resolution sprites" | – | – | **Cannot run.** We would have to re-implement it from the paper. |
| Generating Pixel Art Character Sprites using GANs | F. Coutinho, L. Chaimowicz; SBGames 2022 | arXiv:2208.06413 **[V]** | Author's GitHub (fegemo) has related code; no weights for this paper found | Pose-to-pose image translation, not text | – | – | Not text-to-image, so not comparable |
| A Missing Data Imputation GAN for Character Sprite Generation (MDIGAN) | F. Coutinho, L. Chaimowicz; SBGames 2024 | arXiv:2409.10721 **[V]**; code github.com/fegemo/mdigan-characters **[V]** | Code yes; **no pretrained weights** (you train with train.py) | 64×64 default, configurable; supports RGBA | Yes (4-channel) | – | Needs training; not text-conditioned |
| On the Challenges of Generating Pixel Art Character Sprites Using GANs | Coutinho, Chaimowicz; AIIDE 2022 | https://ojs.aaai.org/index.php/AIIDE/article/view/21951 **[V-search]** | – | Not text | – | – | Not comparable |
| Pixel art character generation as an image-to-image translation problem using GANs | Coutinho, Chaimowicz; Graphical Models 2024 (ScienceDirect S1524070324000018) | **[V-search]** (fetch returned 403) | – | Not text | – | – | Not comparable |
| Pixel VQ-VAEs for Improved Pixel Art Representation | A. Saravanan, M. Guzdial; EXAG 2022 | arXiv:2203.12130 **[V]**; github.com/akashsara/fusion-dance **[V-search]** | Partial | A representation model, not text-to-image | – | – | Not a T2I baseline. Relevant only as related work on pixel-level tokenizers. |
| Guiding generation of 2D pixel art characters using text-image similarity models (MSc thesis) | P. Löwenström; KTH 2024 | https://kth.diva-portal.org/smash/get/diva2:1853314/FULLTEXT02.pdf **[V]** (PDF read) | No code link in the thesis | CLIP-guided diffusion and PixelDraw on humanoid characters | – | – | Cannot run. Cite as related work only. |
| Pixel Art Diffusion (v1–v3.1) | KaliYuga (community); 2022 | github.com/KaliYuga-ai/Pixel-Art-Diffusion **[V]** (repo exists; model details **[U]**) | Colab notebooks (Disco Diffusion forks). Search snippets describe an unconditional model fine-tuned from OpenAI's diffusion model on about 4k 256-px pixel-art landscapes/portraits, steered by CLIP. | No, 256 px scenes | No | – | Old Disco Diffusion stack; outputs are scenes, not sprites |
| Sprite Sheet Diffusion | C.-A. Hsieh, J. Zhang, A. Yan; arXiv 2024 (rev. 2025) | arXiv:2412.03685 **[V]**; github.com/chenganhsieh/Sprite-Sheet-Diffusion **[V]** | Code (MIT) + fine-tuned weights (Google Drive) | Needs a reference character image plus a pose sequence | – | – | Not text-to-sprite; see category 4 |

**Note for the paper on PixDiff-PIG.** Its abstract describes a curated **42k-sprite** dataset, **BLIP captions**, palette quantization and OKLab colour. Search snippets also report FID 19.6 against StyleGAN2 41.7 and LDM 28.4. That setup is very close to ours, so a reviewer who knows it will expect us to discuss it. We cannot run it (no code), but we should cite it and explain the difference. Pin down its authors and venue before citing; I could only reach it through ResearchGate snippets.

### 1b. Open fine-tunes / LoRAs of general T2I models (Hugging Face, Civitai)

These are the de facto "pixel-art-specific" open models. **None generates 12–24 px directly.** All of them output a 512–1024 px image whose pixel grid has to be recovered and downscaled.

| Model | Base | Creator, year | Link | License | Native size / downscale advice | Alpha | Popularity (at fetch time) | How hard to run |
|---|---|---|---|---|---|---|---|---|
| **Pixel Art XL** | SDXL 1.0 LoRA | nerijs, 2023 | HF `nerijs/pixel-art-xl` **[V]** | CreativeML OpenRAIL-M | 1024 px. The card says to *"downscale 8 times … nearest neighbors"* (so a 128 grid). No trigger word. The card recommends a fixed VAE and says not to use the refiner. | No | **15.3k downloads/month**, the most-used pixel-art LoRA | pip diffusers + SDXL base (~7 GB fp16) + LoRA |
| Pixel Art Medium 128 v0.1 | SD3-Medium LoRA | nerijs, 2024 | HF `nerijs/pixel-art-medium-128-v0.1` **[V]** | **CC-BY-NC-4.0** | "Outputs 128x128 pixel art, grid-aligned images" (literal 128 px or 1024 px with 8-px cells is not stated **[U]**). Trigger: "pixel art style". 28 steps, CFG 5. | No | 121/month | SD3-Medium weights (gated) + LoRA |
| Pixel Art 3.5L | SD3.5-Large LoRA | nerijs, 2024 | HF `nerijs/pixel-art-3.5L` **[V]** | Apache-2.0 (LoRA); base SD3.5L has its own licence | "pixel perfect and grid aligned 128x128 outputs". Best checkpoint `pixel-art-3.5L-v2_000000300.safetensors` (147 MB). Trigger: "pixel art". Training data is the `nerijs/pixelparti-128-v0.1` dataset **[V]**: about 4.8k synthetic 128-px images made with Pixel Art XL. | No | 120/month | SD3.5-Large (~16 GB) + LoRA; fits on an A100 |
| PixelArtRedmond | SDXL LoRA | artificialguybr, 2023 | HF `artificialguybr/PixelArtRedmond` **[V]** | CreativeML OpenRAIL-M | 1024 px; triggers "Pixel Art", "PixArFK" | No | 1.5k/month | Same as Pixel Art XL |
| Pixel Art Diffusion XL – Sprite Shaper | SDXL full checkpoint | Yamer, 2024 | civitai.com/models/277680 **[V]** | OpenRAIL++-M **plus creator restrictions** (no redistribution off Civitai, no selling outputs without consent) | 1024 px | No | 27k downloads, 1.8k ratings | 6.46 GB checkpoint from Civitai |
| All-In-One-Pixel-Model | SD 1.x DreamBooth | PublicPrompts, 2022 | HF `PublicPrompts/All-In-One-Pixel-Model` **[V]** | CreativeML OpenRAIL-M | Trigger "pixelsprite" (sprites) / "16bitscene". The card says it is "not pixel perfect". | No | 1.9k/month | 2.1 GB ckpt |
| Modern Pixel art LoRA | FLUX.1-dev LoRA | UmeAiRT | HF `UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art` **[V]** | MIT (LoRA); FLUX.1-dev is non-commercial | 1024 px; trigger "umempart"; 100 training images | No | 1.3k/month | FLUX.1-dev (~24 GB bf16) + LoRA |
| Flux-2D-Game-Assets-LoRA | FLUX.1-dev LoRA | gokaygokay | HF `gokaygokay/Flux-2D-Game-Assets-LoRA` **[V]** | Apache-2.0 (LoRA) | Pixel-art game assets on a **white background**; trigger "GRPZA" | No (white bg) | 538/month | FLUX.1-dev + LoRA |
| Retro-Pixel-Flux-LoRA | FLUX.1-dev LoRA | prithivMLmods | HF `prithivMLmods/Retro-Pixel-Flux-LoRA` **[V]** | OpenRAIL-M | 1024 px; only 16 training images; the author says it is still experimental | No | 228/month | FLUX.1-dev + LoRA |
| Pokémon trainer sprites pixelart | FLUX.1-dev LoRA | sWizad | HF `sWizad/pokemon-trainer-sprites-pixelart-flux` **[V]** | Bespoke licence | Trained on 96×96 sprites + BLIP captions; white background | No | 215/month | FLUX.1-dev + LoRA; narrow domain |
| flux-pixel-art-lora-v1 | FLUX LoRA | matanby | HF `matanby/flux-pixel-art-lora-v1` **[V]** (page exists; **card empty**) | – | – | – | – | Not usable without documentation |
| **Pixel Art Sprite LoRA** | FLUX.2-klein-4B LoRA | Limbicnation, 2026 | HF `Limbicnation/pixel-art-lora` **[V]**; base `black-forest-labs/FLUX.2-klein-4B` **[V]** (Apache-2.0, not gated, ~13 GB VRAM) | Apache-2.0 | 512 px, 4 steps, CFG 1.0; trigger "pixel art sprite, …, game asset, transparent background". Trained on 500 CC0 + synthetic images. | The card claims **"512x512 RGBA output with transparent backgrounds"**. The mechanism is not documented **[U]**: the FLUX VAE is RGB, so alpha is probably added by post-processing. Test it. | 4.5k/month | pip diffusers ≥ 0.37 + 4B base; seconds per image |
| elusarca pixel art style LoRA | Z-Image Turbo LoRA | reverentelusarca | HF `reverentelusarca/elusarca-pixel-art-style-lora-zimage-turbo` **[V]** | Apache-2.0 | Prompt with "pixel art" | No | not tracked | Z-Image Turbo + LoRA |
| *(enabler)* **LayerDiffuse**, transparent latent diffusion | SDXL (SD1.5 planned in the CLI) | L. Zhang, M. Agrawala; 2024 | arXiv:2402.17113 **[V]**; github.com/layerdiffusion/LayerDiffuse **[V]**; weights HF `LayerDiffusion/layerdiffusion-v1` **[V]**; diffusers CLI github.com/lllyasviel/LayerDiffuse_DiffusersCLI **[V]** | Apache-2.0 (code), OpenRAIL-M (weights) | Native RGBA at 1024 | **Yes, native** | – | The diffusers CLI supports only the built-in SDXL. Using it with a pixel-art LoRA probably needs the Forge/ComfyUI route **[U]**. |

---

## 2. Optimisation-based methods

| Method | Authors, venue, year | Link | Code / weights | Native 16–24 px? | Alpha | Cost per image | How hard to run |
|---|---|---|---|---|---|---|---|
| **SD-πXL**: Generating Low-Resolution Quantized Imagery via Score Distillation | A. Binninger, O. Sorkine-Hornung; **SIGGRAPH Asia 2024** | arXiv:2410.06236 **[V]**; github.com/AlexandreBinninger/SD-piXL **[V]** (MIT, 66★) | Yes. Uses SDXL or SSD-1B as the SDS teacher, with optional ControlNet and LoRA. | **Yes.** Output size and palette are set in the config; text-only mode is supported. | Not supported (not mentioned) | README: "a few hours". **Our measurement: about 8 GPU-hours per sprite.** 24 GB VRAM recommended. | Clone + conda (Py 3.10) + SDXL download. Already cloned on our server (see experiment_log.md). |
| **Pixray / PixelDraw** (CLIP-guided pixel drawer) | Tom White (dribnet) et al.; open source, 2021 | github.com/pixray/pixray **[V]** (about 1k★); `pixeldrawer.py` **[V]**; hosted at replicate.com/dribnet/pixray-pixel **[V]** | Yes (CLIP; no training) | **Yes.** `--pixel_size 16 16` sets the grid exactly (defaults are 40×40 square or 80×45 landscape). Colours are free RGB, not a palette. | **Yes.** The renderer keeps an RGBA cell colour and has a transparency mode. | Replicate: about $0.10 per run, "within 8 minutes" on a T4 | Clone with submodules + pip requirements (old 2021 stack; conda recommended) |
| Löwenström thesis (CLIP-guided PixelDraw vs diffusion) | see 1a | see 1a | No | – | – | – | Cannot run |

SD-πXL is the only peer-reviewed, graphics-venue, text-to-pixel-art method with public code. Pixray is the best-known cheap method that also writes directly onto a pixel grid.

---

## 3. Image-to-pixel-art conversion (chain after a T2I model)

| Method | Authors, venue, year | Link | Code / weights | Works to 16–24 px? | Alpha | Cost | How hard to run |
|---|---|---|---|---|---|---|---|
| Pixelated Image Abstraction | T. Gerstner, D. DeCarlo, M. Alexa, A. Finkelstein, Y. Gingold, A. Nealen; NPAR 2012 | gfx.cs.princeton.edu/pubs/Gerstner_2012_PIA **[V]** (no code on the page); journal version cragl.cs.gmu.edu/pixelate **[U]** (TLS error) | **No official code.** Unofficial Rust port github.com/AlexandreBinninger/pixelization **[V]** (1★, v0.1.1) | Yes (explicit output size and palette) | – | CPU | Cargo install of an unofficial port; would need validating |
| Content-Adaptive Image Downscaling | J. Kopf, A. Shamir, P. Peers; SIGGRAPH Asia 2013 (TOG 32(6)) | paper PDF on cs.wm.edu **[V-search]** | Unofficial MATLAB github.com/AyushRai09/Content-Adaptive-Image-Downsampling **[V]** (23★; README hints at a small-input limit, so probably partial) | Yes | – | CPU | Needs MATLAB and validation. Low priority. |
| Deep Unsupervised Pixelization | C. Han, Q. Wen, S. He, Q. Zhu, Y. Tan, G. Han, T.-T. Wong; SIGGRAPH Asia 2018 (TOG 37(6)) | project page ttwong12.github.io/papers/pixel/pixel.html **[V]** (still says "source code will be released soon") | Mirror github.com/PeterZs/Deep-Unsupervised-Pixelization **[V]**: training code only, **no pretrained weights**, PyTorch 0.4 / Ubuntu 16.04 | Designed for moderate cell sizes | – | – | **Needs training** on an old stack |
| **Make Your Own Sprites**: Aliasing-Aware and Cell-Controllable Pixelization | Z. Wu et al.; SIGGRAPH Asia 2022 (TOG 41(6)) | github.com/WuZongWei6/Pixelization **[V]** (441★) | **Weights yes** (4 checkpoints via Google Drive). **Licence: non-commercial research only.** | The "Test Pro" mode takes any integer `--cell_size`. A 1024→16 reduction means 64-px cells, which is far outside its training range. | – | GPU, seconds | Linux + PyTorch ≥ 1.7 + Drive download |
| **PixelOE**: Detail-Oriented Pixelization based on Contrast-Aware Outline Expansion | Shih-Ying Yeh (KohakuBlueleaf); 2024, no paper | github.com/KohakuBlueleaf/PixelOE **[V]** (491★, Apache-2.0) | `pip install pixeloe` | Yes. Modes: center, contrast, k-centroid, bicubic, nearest; optional palette quantization. | Not mentioned (apply the mask separately) | More than 180 img/s on an RTX 4090 | pip, trivial |
| Super Pyxelate | sedthh | github.com/sedthh/pyxelate **[V]** (1.7k★, MIT) | pip (from git) | Yes. HOG downsampling + Bayesian-GMM palette. | **Yes** (alpha threshold) | CPU | pip, trivial |
| Pixel Art Fixer | Retro Diffusion team | github.com/Retro-Diffusion/pixel-art-fixer **[V]** (393★, MIT) | Python / Rust | Recovers the true grid of *fake* pixel art. It does not reduce to an arbitrary target size. | **Yes** (majority per cell) | CPU | pip deps |
| proper-pixel-art | K. J. Allen | github.com/KennethJAllen/proper-pixel-art **[V]** (521★, MIT) | `pip install proper-pixel-art` | Grid recovery for AI "pixel-art-style" images (it names gpt-image-2 outputs explicitly) | Yes (`--transparent`) | CPU | pip, trivial |
| FLUX.1-Kontext Pixel-Style LoRA (image → pixel-art edit) | Shakker-Labs | HF `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Pixel-Style` **[V]** | FLUX.1 non-commercial licence | High-res output | No | – | Kontext-dev + LoRA |
| SD-πXL, image-conditioned mode | see §2 | see §2 | yes | yes | no | hours | see §2 |
| *(alpha helpers)* BiRefNet; rembg | Zheng Peng et al.; D. Gatis | HF `ZhengPeng7/BiRefNet` **[V]** (MIT, 0.2B); github.com/danielgatis/rembg **[V]** (24.7k★, MIT) | yes | – | produce the mask | – | pip |

---

## 4. Sprite-sheet / game-asset generators

| Method | Base / type | Link | Licence | Size | Alpha | How hard to run |
|---|---|---|---|---|---|---|
| SD_PixelArt_SpriteSheet_Generator (same model as Civitai "Pixel Art Sprite Diffusion" #22) | SD 1.5 checkpoint, 4 views (triggers PixelartFSS/BSS/LSS/RSS) | HF `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` **[V]**; civitai.com/models/22 **[V]** | Apache-2.0 (HF) / OpenRAIL-M (Civitai) | 512-class, downscale needed | No (the author suggests removing the background by hand) | pip diffusers, 4 GB |
| pixel_spritesheet_4walk_small_lora_v1 | FLUX.2-klein-**base**-4B LoRA | HF `svntax-dev/pixel_spritesheet_4walk_small_lora_v1` **[V]** | Apache-2.0 | 512 output of **32×32 characters**; the card says downscale ×4 with k-centroid | – | FLUX.2-klein-base + LoRA |
| flux-lora-spritesheet (LPC style) | FLUX.2-klein-base-9B LoRA | HF `Mystic07/flux-lora-spritesheet` **[V]** | non-commercial | 768–1024 px sheets; about 18.7k LPC training sheets | – | 9B base + LoRA |
| Sprite Sheet Diffusion | Animate-Anyone fine-tune | arXiv:2412.03685 **[V]**; GitHub **[V]** | MIT | needs a reference image + poses | – | Weights on Drive; not T2I |
| MDIGAN | GAN | see 1a | – | 64×64 RGBA | yes | needs training; not T2I |
| pixel-forge | 1M–17M-param EDM diffusion, **class-conditioned**, pure Rust | github.com/cochranblock/pixel-forge **[V]** (10★, Unlicense) | public domain | 32×32 | – | Obscure; not open-vocabulary text |
| sprite-diffusion | unconditional DDPM on 1,788 16-px sprites | github.com/oelin/sprite-diffusion **[V]** (0★, MIT) | MIT | **16×16** | – | Toy; no text conditioning |
| Pixel Art Bench (LLMs emit 24×24 JSON grids) | evaluation harness | huggingface.co/blog/AINovice2005/pixel-art-bench **[V]** (May 2026); code github.com/ParagEkbote/pixel-art-bench | – | **24×24 native** | could be | Possible zero-shot "LLM draws the grid" baseline. Cheap but weak. |

---

## 5. Commercial / API pixel-art generators

| Service | Link | Native 12–24 px? | Alpha | Price per image | How hard to run |
|---|---|---|---|---|---|
| **Retro Diffusion API** (Astropulse) | API docs github.com/Retro-Diffusion/api-examples **[V]**; also replicate.com/retro-diffusion/rd-fast **[V]**; resold via Scenario (scenario.com/models/retro-diffusion-plus **[V]**) | **Yes.** `rd_pro__{default,simple,fantasy,scifi,platformer,…}` accept **12–256 px**. `rd_pro__pixelate` accepts 16–256. Low-res styles (`rd_plus__low_res`, `rd_fast__low_res`, `rd_mini__skill_icon`, `rd_mini__topdown_item`, …) are listed as "smaller" than 64; the style page says 16–512 **[V]**, so exact minimum **[U]**. | **Yes:** `"remove_bg": true` | rd_pro **$0.18**. Low-res styles `max(0.02,(w·h+13700)/600000)` ≈ **$0.023** at 16² and $0.024 at 24². rd_plus `max(0.025,(w·h+50000)/2e6)`. rd_fast about $0.015. | REST API: `POST https://api.retrodiffusion.ai/v2/inferences`, header `X-RD-Token`, poll `/tasks/{id}`, images come back in `base64_images`. Batch ≤ 16. |
| Retro Diffusion Aseprite extension (local) | astropulse.itch.io/retrodiffusion **[V]** | "any size" | background removal yes | $65 one-off (Lite $20) | Runs locally but the model is proprietary. The page warns it is **not** the same model as the website/API. |
| **PixelLab API** | pixellab.ai/pixellab-api **[V]**; Python SDK github.com/pixellab-code/pixellab-python **[V]** (`pip install pixellab`) | **Partly.** "Create S-XL image (Pro)" minimum is **16×16** **[V]**, so 16/20/24 work and 12 does not. Pixflux has a minimum *area* of 32×32 (a 16×16 canvas is rejected; 16×64 is accepted). Bitforge price table starts at 32×32. | **Yes:** "No Background" option; transparent prices listed | Pixflux $0.0079 (64²); Bitforge $0.0071 (32²), transparent $0.0073; **Pro up to 256²: $0.095**. The UI says ≤ 32 px Pro calls return 64 variations; whether the API does the same is **[U]**. | API key + SDK. The exact Pro endpoint name is **[U]** (API docs page is JS-rendered). |
| Recraft API (`digital_illustration` / `pixel_art` substyle) | recraft.ai/docs/api-reference/styles **[V]**; pricing recraft.ai/docs/api-reference/pricing **[V]** | No: a general model with a pixel-art style, V2/V3 only; outputs are about 1024 px | Separate background-removal endpoint (cost **[U]**) | Raster $0.022–$0.25 | REST API |
| Sprite AI | sprite-ai.art **[V]** | Vendor blog claims native 32–512 px | yes | subscription, $7–$42/month | API/MCP |
| SpriteLab | spritelab.dev/docs/api **[V]** | `/micro` produces exact-size icons down to 16×16 | yes | credits (no USD price listed) | REST API; underlying model not disclosed |
| Scenario | scenario.com **[V]** | hosts Retro Diffusion Plus + custom fine-tunes | bg removal tool | – | API |
| Others (DeepAI, Leonardo, Pixie.haus, OpenArt, getimg, starryai, Adobe Firefly, …) | listed at github.com/equally-creator/awesome-ai-pixel-art-generator **[V]** (list only; tools not checked individually) | Mostly general models with a pixel style at about 1024 px | – | – | Not worth benchmarking |

---

## 6. Excluded after checking (not pixel-art methods or out of scope)

- *Palette Aligned Image Diffusion* (arXiv:2509.02000 **[V]**): palette-conditioned T2I, not pixel art.
- PixelDiT, PixelFlow, DiP, PixelGen (arXiv 2511.20645, 2504.07963, 2511.18822, 2602.02493): "pixel-space diffusion" means no VAE. They are not about pixel art.
- *Voxify3D* (arXiv:2512.07834): 3D voxel art.
- *GameTileNet* (arXiv:2507.02941): a dataset, not a generator.
- Pre-2020 work that belongs in related work only: Serpa & Rodrigues (SBGames 2019); Jiang & Sweetser, GAN-assisted YUV pixel art (AJCAI 2021; Springer link redirected to login, **[U]**).

## 7. Could not verify (flag before citing)

1. **PixDiff-PIG**: authors, venue, year, and whether code exists (ResearchGate 403; no other copy found).
2. **Deep Unsupervised Pixelization**: no official code or weights. The PeterZs repo is a mirror, not confirmed official.
3. **Gerstner 2012**: no official code; the GMU journal page failed TLS.
4. **nerijs 128 LoRAs**: whether the output is a literal 128 px image or 1024 px with 8-px cells.
5. **Limbicnation/pixel-art-lora**: how the claimed RGBA output is produced.
6. **LayerDiffuse + third-party SDXL LoRA** in diffusers: untested; the CLI documents only the base SDXL.
7. **Retro Diffusion low-res styles**: exact minimum size (12 px is confirmed only for `rd_pro__*`).
8. **PixelLab Pro API**: endpoint name, and variations returned per call at ≤ 32 px.
9. KaliYuga Pixel Art Diffusion model details; Coutinho 2024 journal page; Kopf 2013 PDF (search-only).

---

## 8. Ranked shortlist: baselines that most strengthen a SOTA claim

Ranking criteria: (a) designed for pixel art, (b) public weights runnable on one A100 80 GB, (c) something a reviewer would expect. Commercial APIs fail (b), but a reviewer asking for pixel-art-specific competitors will expect one. Include exactly one so we are not compared only against open hobby LoRAs.

### #1 SD-πXL (Binninger & Sorkine-Hornung, SIGGRAPH Asia 2024)
- Why: the only peer-reviewed text-to-pixel-art method at a graphics venue with public code. It optimises directly at the target grid and palette. Any reviewer in this area will expect it. Its roughly 8 GPU-hours per sprite is itself evidence for our method (thousands of times faster), so report it.
- What to download and run:
  - `git clone https://github.com/AlexandreBinninger/SD-piXL`, then a conda env with Python 3.10 and `pip install -r requirements`. The repo is already cloned on the server (see experiment_log.md: `/mnt/data/kw/RoundSquisheen/texture/SD-piXL` with conda env `SD-piXL`).
  - Teacher weights from HF: `stabilityai/stable-diffusion-xl-base-1.0` **[V]** (OpenRAIL++-M, not gated). Optionally add the `nerijs/pixel-art-xl` LoRA as the style teacher, which the repo supports.
  - Config: set the output to 12/16/20/24 and a palette (for example, the palette size our method uses). Prompt with "…, on a plain white background". Alpha: key out the background colour, or run BiRefNet on a nearest-×32 upscale and threshold. State this in the paper, because SD-πXL has no alpha.
  - Budget: a 30–50-prompt subset at 16 and 24 px (about 8 h × 100 ≈ 800 A100-hours at our measured rate; cut to 16 px only if needed). Per MEMORY, run on node09 inside tmux.

### #2 SDXL + Pixel Art XL LoRA (`nerijs/pixel-art-xl`), with proper grid recovery and alpha
- Why: the most downloaded open pixel-art model (15.3k/month), the one practitioners actually use, and a direct answer to "your SDXL baseline isn't pixel-art-tuned". It uses the same backbone as our current SDXL baseline, so the comparison isolates the pixel-art fine-tune.
- What to download and run:
  - `pip install diffusers transformers accelerate`. HF: `stabilityai/stable-diffusion-xl-base-1.0` **[V]**, `madebyollin/sdxl-vae-fp16-fix` **[V]** (the card's "fixed VAE"), `nerijs/pixel-art-xl` **[V]**.
  - Generate 1024² once per prompt (no refiner). Estimate: about 3–5 s per image on an A100 at 30 steps, so the 3000-prompt held-out set takes about 3–4 h.
  - Post-process: nearest ÷8 to 128 (per the card) → downscale to 12/16/20/24 with **the same downscaler protocol used for gpt-image-2/SDXL**. Also report PixelOE (`pip install pixeloe`) and pick the best downscaler per baseline, so nobody can say we chose a bad one. Alpha: BiRefNet (`ZhengPeng7/BiRefNet` **[V]**) mask at 1024, area-downscaled and thresholded at 0.5.
  - Optional variant: swap the LoRA for `nerijs/pixel-art-3.5L` (SD3.5-Large, Apache LoRA, grid-aligned 128) to show the finding holds on a newer backbone.

### #3 Retro Diffusion API (commercial, pixel-art-native, 12–256 px, `remove_bg`)
- Why: the strongest commercial pixel-art-specific generator. Its models were trained with pixel artists on licensed art, and it is **the only candidate that natively outputs 12–24 px with transparency**, exactly our task format. It is widely known among game developers and cheap enough to run on the full test set with the low-res styles.
- What to run:
  - Create an account at retrodiffusion.ai and get an API key (`rdpk-…`).
  - `POST https://api.retrodiffusion.ai/v2/inferences` with header `X-RD-Token`, body `{"prompt": …, "prompt_style": "rd_pro__default" | "rd_pro__simple", "width": 16, "height": 16, "num_images": 1, "remove_bg": true, "seed": s}`. Poll `GET /v2/inferences/tasks/{task_id}` and decode `result.base64_images`.
  - Cost: rd_pro is $0.18/image, so 200 prompts × 4 sizes ≈ $144. `rd_plus__low_res` is about $0.023/image, so the full 3000 × 4 sizes ≈ $280 (check whether it accepts 12 px; if not, use rd_pro for 12).
  - Freeze the date, style names and seeds in the paper, since the service changes over time.

### #4 T2I + learned pixelization chain: gpt-image-2 / SDXL outputs → Make Your Own Sprites (Wu et al., SIGGRAPH Asia 2022) and PixelOE
- Why: this answers "you only used naive downscaling" with the academic state of the art in image-to-pixel-art (TOG 2022, 441★) plus the popular training-free PixelOE. It is nearly free because the 1024-px gpt-image-2 and SDXL outputs already exist.
- What to download and run:
  - `git clone https://github.com/WuZongWei6/Pixelization`. Download the 4 checkpoints (structure extractor, AliasNet, I2PNet, P2INet) from the Google Drive links in the README. Linux + PyTorch ≥ 1.7. Licence: non-commercial research only, which is fine for the paper.
  - Run the "Test Pro" mode with `--cell_size` set so the cell grid lands at or near the target size. For extreme ratios, first nearest-downscale to about 4–8× the target, then pixelize, then box-reduce. Report this protocol.
  - `pip install pixeloe` and run on the same inputs. Alpha comes from the same BiRefNet mask as in #2.

### #5 PixelLab API (commercial, pixel-art-native, 16–512 px, transparent background)
- Why: the other widely known pixel-art-native commercial tool. Add it if a second commercial competitor is wanted, or use it in place of #3.
- What to run: `pip install pixellab` and get an API key. Use the "Create S-XL image (Pro)" tool at 16/20/24 with "No Background" ($0.095 per call up to 256²). **It cannot do 12 px** (minimum 16), so mark 12 px as N/A. Pixflux and Bitforge reject canvases under a 32×32 area.

### Alternates (open, cheap, native-grid; include if space allows)
- **Pixray PixelDraw**: `git clone --recursive https://github.com/pixray/pixray`, then `python pixray.py --drawer=pixel --pixel_size 16 16 --prompt "…" [transparency on]`. It writes directly onto a 16×16 RGBA grid with CLIP guidance, taking minutes per sprite. It is a far cheaper optimisation baseline than SD-πXL and can cover the full test set, but it uses 2021-era CLIP guidance.
- **FLUX.2-klein-4B + `Limbicnation/pixel-art-lora`**: the newest open pixel-sprite LoRA (2026), Apache-2.0, 4-step generation, trained for "transparent background" game sprites. This is the "modern open fine-tune" point if a reviewer asks about post-SDXL models. Check how its alpha is produced first.

### Fairness protocol to state in the paper (applies to #2, #4 and the existing gpt-image-2/SDXL baselines)
1. Use the same prompt set and the same post-processing for every 1024-px baseline. Report both the "native" downscaler and the best one out of {nearest-from-grid, box, PixelOE, k-centroid}.
2. Compute alpha the same way for all baselines without native alpha (BiRefNet mask → area-downscale → threshold 0.5). For methods with native alpha (Retro Diffusion `remove_bg`, PixelLab no-bg, pixray transparency), use their own alpha.
3. Report wall-clock and cost per sprite next to quality. SD-πXL at about 8 GPU-hours and the APIs at $0.02–0.18 per image are part of the argument.
