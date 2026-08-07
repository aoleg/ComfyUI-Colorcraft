# Colorcraft

**Color grading for ComfyUI, applied where it actually matters: inside the diffusion process itself.**

Colorcraft is a set of modular nodes that chain together, letting you build your own custom color-editing pipeline out of modifiers, schedules, and masks. Letting you mix and match exactly what a shot needs. 

Between the axes and masks on offer, it's close to the full toolset of something like Lightroom or Camera Raw... applied somewhere those guys could never reach: mid-generation, in latent space.

Exposure, Contrast, Tone Mapping, Saturation et al., White Balance/Tint, Hue Shifting, Split-toning and Cross-processing effects, and specifically targeted versions of those, all done without LoRAs, prompt hijinks, CFG boost, post-processing effects etc., just using pure vector math on the latent.

And since it's just simple vector math, **it's computationally close to free!**

<!-- TODO: hero image — assets/hero.png -->
<!-- ![Colorcraft hero](assets/hero.png) -->

## Why not just do it in post?

Most color tools work on a finished image after the fact: a curve, a LUT, a filter laid over pixels that are already locked in. Colorcraft reaches into the latent while the image is still being formed, and shapes color the same way the model itself does, along real, meaningful axes of the space it thinks in, rather than the red/green/blue channels of a decoded image. 

It's the difference between metering a shot correctly at capture versus fixing the exposure afterward, or mixing the right color on the palette versus color-correcting a finished painting. Latents carry far more dynamic range than a decoded image, so there's real headroom to work with: proper HDR-range color, not a clipped approximation of it. 

And because the edit happens while the image is still forming, it doesn't just recolor the result: it can steer the generation itself (e.g. making darker, brighter, or more colorful compositions than the model would produce on its own). You can even push or pull fine detail and texture directly, something no post-process filter can genuinely add back once it's gone.

Oh, and also, you won't need a second software or process. You do it all in one go.

## What you get

- **Basic** — contrast and color shift, works on any model
- **Advanced** — the full toolkit in one mega-node
- **Luma** — exposure, tone compression
- **Chroma** — temperature, tint, vibrance, saturation, chroma contrast
- **Chroma Plus** — the finer diagonal color axes, for when temperature/tint isn't enough
- **Punch** — contrast, clarity, sharpness
- **Shift** — push/pull colors towards a specifically defined color
- **Schedule** — build one schedule and share it across several modifiers, or use different schedules for different modifiers
- **Masking** — key any edit by color, luminance, or hue
- **Mask Blur** — blur masks and control the spread (grow/shrink)
- **Combine Masks / Mask Blur** — build up complex, compound masks from simple ones
- **Mask Preview** — tiny helper node for visualizing masks
- **Sampler** — the actual workhorse. Chain together whatever modifiers you want, feed the last one into this, and pass it to a SamplerCustom node in place of your regular sampler

<!-- TODO: gallery — assets/gallery/ -->
<!-- before/after comparisons -->

## Requirements

- ComfyUI
- **Basic** node should work with any model that has a VAE, no restrictions
- Every other node needs a matching basis for the model's VAE family. Currently supported:
  - **Krea2** / **Qwen Image** / etc.
  - **Z-Image** / **Flux** / etc.

Colorcraft's color axes are derived per VAE family, not per model, so any model sharing one of the VAEs above is covered automatically (that's the "etc."). 

That being said, I have only tested Krea2 and Z-Image. So, feedback would be much appreciated on how it fares with **Anima**, **Qwen Image**, and other models.

Support for other models (**Flux2**?) might be coming along.

## Installation

**Via ComfyUI Manager:** 
Open Manager → **Install via Git URL** → paste `https://github.com/muerrilla/ComfyUI-Colorcraft` → Confirm.

**Manually:**
```
cd ComfyUI/custom_nodes
git clone https://github.com/muerrilla/ComfyUI-Colorcraft.git
```

Restart ComfyUI and refresh your browser. The nodes appear under **Muerrilla → Colorcraft** in the node menu.

## Getting started

Colorcraft works through a custom sampler. So drop in the `Colorcraft Sampler` node, plug in the required VAE and sampler (`KSamplerSelect`), and plug it into a `SamplerCustom` node. 

Build out from there, chaining together the modifiers (e.g. `Colorcraft Luma` for adjusting exposure) of your liking and feeding them into the `Colorcraft Sampler` node.

Every modifier node (except `Colorcraft Basic` and `Colorcraft Advanced`) requires a `Colorcraft Schedule` input, defining the sampling steps to which the modifier applies.

Additionally, every modifier node (except `Colorcraft Basic`) can take an optional mask through the `masking` input.

## Workflows

A handful of basic example workflows are included in `workflows/` folder. Drop any of them straight into ComfyUI to get oriented. A wider set covering the more advanced adjustments and masking/scheduling tricks is coming shortly.

**Note:** The example workflows are all based on Krea2. You can change the Diffusion Model, CLIP, and VAE to make them work with Z-Image or any other model that uses either Qwen Image VAE or Flux AE. You might have to re-adjust the modifier sliders to get the same adjustment intensities for different models.

## Tips

- **Timing is key:** every adjustment rides its own schedule across the sampling steps. Push it early and it steers the generation; hold it for the later steps and it behaves like straightforward color correction, faithful to what was already forming.

- **Be gentle, or give the model time to heal:** If your edits are too strong, they will eventually break the latent and create artifacts. In those cases you'd better spread the edit across multiple steps and/or avoid applying the edit on the last few steps, giving the model some time to recover. **In general, try to avoid applying edits on the very last step (unless the edit is *very* mild).**

- **Every adjustment can be gated by a mask:** not a hand-painted region, but a live read of the image's own color, luminance, or hue. So, besides things like adjusting only the shadows or highlights, an edit can target "the warm highlights" or "everything except skin tones" using combined masks.

## License

Colorcraft is released under the **GNU GPLv3**. Use it, modify it, build on it freely, including commercially, but if you distribute a modified version, it has to stay open under the same license.

## Credits

The sigma-to-step handling is adapted from Jonseed's ComfyUI port of Detail Daemon: https://github.com/Jonseed/ComfyUI-Detail-Daemon
