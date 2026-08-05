# Colorcraft

**Color grading for ComfyUI, applied where it actually matters: inside the diffusion process itself.**

<!-- TODO: hero image — assets/hero.png -->
<!-- ![Colorcraft hero](assets/hero.png) -->

Most color tools work on a finished image after the fact — a curve, a LUT, a filter laid over pixels that are already locked in. Colorcraft works differently: it reaches into the latent while the image is still being formed, and shapes color the same way the model itself does — along real, meaningful axes of the space it thinks in, rather than the red/green/blue channels of a decoded photo.

It's the difference between metering a shot correctly at capture versus fixing the exposure afterward — or mixing the right color on the palette versus color-correcting a finished painting. Latents carry far more dynamic range than a decoded image, so there's real headroom to work with: proper HDR-range color, not a clipped approximation of it. And because the edit happens while the image is still forming, it doesn't just recolor the result — it can steer the generation itself, pushing a render darker or brighter than the model would produce on its own, or reworking how a scene resolves. You can even push or pull fine detail and texture directly, something no post-process filter can genuinely add back once it's gone.

Timing is part of the tool, not an afterthought: every adjustment rides its own schedule across the sampling steps. Push it early and it steers the generation itself; hold it for the later steps and it behaves like straightforward color correction, faithful to what was already forming. And since it's just vector math on the latent, it's computationally close to free.

On top of that, every adjustment can be gated by mask — not a hand-painted region, but a live read of the image's own color, luminance, or hue, so an edit can target "the warm highlights" or "everything except skin tones" without ever touching a brush.

## What you get

Colorcraft is a set of modular nodes that chain together, letting you build your own custom color-editing pipeline out of modifiers, schedules, and masks — mix and match exactly what a shot needs. Between the axes on offer, it's close to the full toolset of something like Lightroom or Camera Raw — applied somewhere a raw editor never could reach: mid-generation, in latent space.

- **Basic** — contrast and color shift, works on any model
- **Advanced** — the full toolkit in one node
- **Luma** — exposure, tone compression
- **Chroma** — temperature, tint, vibrance, saturation, chroma contrast
- **Chroma Plus** — the finer diagonal color axes, for when Chroma's basics aren't enough
- **Punch** — contrast, clarity, sharpness
- **Shift** — a dedicated, live-previewed color push/pull
- **Schedule** — build one timing curve and share it across several modules
- **Masking** — gate any edit by color, luminance, or hue
- **Combine Masks / Mask Blur** — build up complex, compound masks from simple ones
- **Sampler** — the actual workhorse. Chain together whatever modifiers you want, feed the last one into this, and pass it to a SamplerCustom node in place of your regular sampler

<!-- TODO: gallery — assets/gallery/ -->
<!-- A handful of before/after comparisons will go here. -->

## Requirements

- ComfyUI
- **Basic** works with any model that has a VAE, no restrictions
- Every other node needs a matching basis for the model's VAE family. Currently supported:
  - **Krea2** / **Qwen Image** / etc.
  - **Z-Image** / **Flux** / etc.

Colorcraft's color axes are derived per VAE family, not per model, so any model sharing one of the VAEs above is covered automatically (see the "etc." — that's the point). Support for additional families is planned.

## Installation

**Via ComfyUI Manager:** open Manager → **Install via Git URL** → paste `https://github.com/muerrilla/ComfyUI-Colorcraft` → Confirm.

**Manually:**
```
cd ComfyUI/custom_nodes
git clone https://github.com/muerrilla/ComfyUI-Colorcraft.git
```

Restart ComfyUI. The nodes appear under **Muerrilla → Colorcraft** in the node menu.

## Getting started

A handful of basic example workflows are included in `workflows/` — drop any of them straight into ComfyUI to get oriented. A wider set covering the more advanced masking/scheduling tricks is coming shortly. In the meantime: start from **Basic** or **Advanced** feeding into **Sampler** in place of your regular sampler — everything else in the pack builds out from there.

## License

Colorcraft is released under the **GNU GPLv3**. Use it, modify it, build on it — freely, including commercially — but if you distribute a modified version, it has to stay open under the same license.

## Credits

The sigma-to-step handling is adapted from Jonseed's ComfyUI port of Detail Daemon: https://github.com/Jonseed/ComfyUI-Detail-Daemon
