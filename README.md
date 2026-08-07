# Colorcraft

### Color grading for ComfyUI, applied where it actually matters: inside the diffusion process itself.

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

![Screenshot of all nodes in the pack](assets/nodes.jpg)
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
- **Combine Masks** — build up complex, compound masks from simple ones
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

### ComfyUI

**Via ComfyUI Manager:** 
Open Manager → **Install via Git URL** → paste `https://github.com/muerrilla/ComfyUI-Colorcraft` → Confirm.

**Manually:**
```
cd ComfyUI/custom_nodes
git clone https://github.com/muerrilla/ComfyUI-Colorcraft.git
```
Restart ComfyUI and refresh your browser. The nodes appear under **Muerrilla → Colorcraft** in the node menu.

### WebUI Forge Neo

The same repo doubles as a Forge Neo (`sd-webui-forge-classic`) extension — see [Forge Neo](#forge-neo) below for what it covers.

```
cd webui/extensions
git clone https://github.com/aoleg/ComfyUI-Colorcraft.git
```
Restart the webui. Colorcraft appears as an accordion on both the txt2img and img2img tabs.

## Getting started

Colorcraft works through a custom sampler. So drop in the `Colorcraft Sampler` node, plug in the required VAE and sampler (`KSamplerSelect`), and plug it into a `SamplerCustom` node. 

Build out from there, chaining together the modifiers of your liking and feeding them into `Colorcraft Sampler`. (e.g. `Colorcraft Luma` for adjusting exposure → `Colorcraft Punch` for adjusting contrast → `Colorcraft Sampler`)

Every modifier node (except `Colorcraft Basic` and `Colorcraft Advanced`) requires a `Colorcraft Schedule` input, defining the sampling steps to which the modifier applies.

Additionally, every modifier node (except `Colorcraft Basic`) can take an optional mask through the `masking` input.

## Workflows

A handful of basic example workflows are included in `workflows/` folder. Drop any of them straight into ComfyUI to get oriented. A wider set covering the more advanced adjustments and masking/scheduling tricks is coming shortly.

**Note:** The example workflows are all based on Krea2. You can change the Diffusion Model, CLIP, and VAE to make them work with Z-Image or any other model that uses either Qwen Image VAE or Flux AE. You might have to re-adjust the modifier sliders to get the same adjustment intensities for different models.

## Tips

- **Timing is key:** every adjustment rides its own schedule across the sampling steps. Push it early and it steers the generation; hold it for the later steps and it behaves like straightforward color correction, faithful to what was already forming.

- **Be gentle, or give the model time to heal:** If your edits are too strong, they will eventually break the latent and create artifacts. In those cases you'd better spread the edit across multiple steps and/or avoid applying the edit on the last few steps, giving the model some time to recover. **In general, try to avoid applying edits on the very last step (unless the edit is *very* mild).**

- **Every adjustment can be gated by a mask:** not a hand-painted region, but a live read of the image's own color, luminance, or hue. So, besides things like adjusting only the shadows or highlights, an edit can target "the warm highlights" or "everything except skin tones" using combined masks.

## Forge Neo

Colorcraft also runs as an extension for **WebUI Forge Neo** (`sd-webui-forge-classic`, neo branch). It is the same math — `lib_colorcraft/` is shared by both frontends, not reimplemented — reached through a flat panel instead of a node graph.

### Where it hooks

`Colorcraft Sampler` outputs a `SAMPLER`, but it never implements one: the wrapper registers a post-CFG function and then calls whatever base solver you gave it. So on Forge there is no sampler wrapper at all — the extension registers the same function via `set_model_sampler_post_cfg_function` on a UNet clone, per pass. Your **Sampling method** and **Schedule type** are untouched.

Three things the node reads off its graph, the extension sources itself: the sigma schedule (from `transformer_options["sampling_sigmas"]`, read lazily on the hook's first call), the latent format (from `p.sd_model.model_config`, since Forge doesn't hang it on the UNet), and the colour anchors — which are VAE-encoded up front, *before* sampling, because a `vae.encode` from inside the sampling loop can evict the UNet.

### What the panel covers

One modifier, one schedule, one mask. That's `Colorcraft Advanced` plus `Masking` — which between them reach every axis in the pack. Controls are tiered exactly as the node tiers them: `Schedule shaping`, `More colors`, `Color shift` and `Advanced` are the node's own `advanced` / `more_colors` / `color_shift` / `dev` booleans, and they gate the same branches of the math.

Not in this phase, because they need graph structure a form can't express:
- **chaining several modifiers** (Luma → Punch → Chroma with different schedules) — the panel is a single modifier;
- **mask trees deeper than two leaves** — you get Mask A, one operation, Mask B;
- **blurring one leaf but not the other** — the blur applies to the combined mask.

`Mask Preview` is folded in as a checkbox rather than a node, so you can tune a mask without rewiring anything.

Two deliberate differences from the node's widgets: `plot_steps` is gone (it only drew tick marks on the LiteGraph plot; the webui already knows the step count), and the ±10 sliders are ±3 here, since a Gradio slider at ±10/0.01 is unusable for the gentle adjustments this is built for. `lib_colorcraft/params.py` records every range change explicitly and a test asserts none of them drifted by accident.

### Settings persistence

Nothing is written to `ui-config.json` — there are ~60 controls and they're per-image settings, not preferences. Your look travels in the PNG instead, as a single compact `Colorcraft:` infotext key holding only what you changed. Send-to-txt2img and paste both restore the whole panel, including resetting anything the pasted look didn't set.

### Model support

Support is per **VAE family**, not per checkpoint — the colour axes are derived from the latent space, so anything sharing a supported VAE is covered:

| Latent format | Models | Vectors |
|---|---|---|
| `Wan21` | Krea 2, Qwen-Image, Anima, Wan 2.1 | `colorcraft-krea2` |
| `Flux` | Flux, Z-Image, Lumina2, Chroma | `colorcraft-zimage` |

Anything else (SD 1.5, SDXL, Flux2) has no basis: **Contrast** and **Color shift** still work, everything else — including masking — silently does nothing, and the log says so once per run. Only Krea 2 and Z-Image have been calibrated; the rest inherit their family's numbers and may want different slider values.

### Verifying a change

`python tests/harness.py` runs the whole port offline in a couple of seconds — no GPU, no model, no webui. It checks four things: that the shared-core refactor changed no values (against the pre-refactor `nodes.py`, recovered from git), that the flat panel builds the same chain a node graph would, that the Forge hook's output matches the ComfyUI node's on identical tensors, and that the parameter table hasn't drifted from `INPUT_TYPES`. Run it before a live generation, not after a suspicious image.

## Limitations

The nodes are designed for the classic UI. Nodes 2.0 is not supported until there's at least some dev docs for it.

## Credits

The sigma-to-step handling is adapted from Jonseed's ComfyUI port of Detail Daemon: https://github.com/Jonseed/ComfyUI-Detail-Daemon
