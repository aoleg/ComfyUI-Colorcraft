# Developer notes

For modifying Colorcraft. If you just want to *use* it, the [README](README.md) is
the whole story.

## Layout

```
nodes.py            ComfyUI node classes
lib_colorcraft/     shared core — every frontend runs this, none reimplements it
  core.py             vector maths, schedules, mask trees, basis loading
  engine.py           the per-step body, framework-agnostic
  params.py           declarative parameter table (data, JSON-dumpable)
  spec.py             flat UI values <-> chain dicts, infotext ser/de
scripts/            the Forge Neo extension
tools/              dev tools, not loaded by anything — see tools/README.md
tests/              offline test suites
vectors/            colorcraft-<family>.safetensors, shared by all frontends
```

`params.py` is deliberately data rather than code: a future SwarmUI frontend is C#
and drives the ComfyUI backend by emitting graph nodes, so it can consume the
parameter table as JSON but not `engine.py`.

## Where the Forge extension hooks

`ColorcraftSampler` returns a ComfyUI `SAMPLER`, which makes it look like a sampler
wrapper. It isn't one: it registers a post-CFG function and then calls whatever base
solver it was handed. Classify a node by what it *reads*, not by its output type.

So on Forge there is no sampler wrapper. The extension registers the same function
via `set_model_sampler_post_cfg_function` on a UNet clone in
`process_before_every_sampling`, the way `modules/processing_scripts/mahiro.py` does.
The user's sampling method and schedule type are untouched. And because Colorcraft is
*already* a post-CFG effect upstream, none of the usual pre-CFG delta reconstruction
applies — the hook reads `denoised` and returns a new `denoised`, exactly as the node
does.

Three things the node gets from its graph, the extension has to source itself:

| Need | Where it comes from |
|---|---|
| the full sigma schedule | `model_options["transformer_options"]["sampling_sigmas"]`, written *after* `process_before_every_sampling` runs — so it is read lazily, on the hook's first call |
| the latent format | `p.sd_model.model_config.latent_format`; Forge does not hang it on the UNet, so `args["model"].latent_format` is always `None` |
| the colour anchors | `vae.encode`, done **up front**. A `vae.encode` from inside the sampling loop reaches `load_models_gpu([vae.patcher])` and can evict the UNet mid-run |

Two more traps worth knowing before touching this file. The prediction tensors handed
to a post-CFG hook are shared with every other post-CFG hook in the list, so every
operation in `core.py` is out-of-place — even the "no-op" branches return the input
object rather than mutating it. And anything that touches VAE weights must run inside
`torch.inference_mode()`, not `no_grad`: Forge loads them in inference mode, and a
plain `no_grad` dies in the manual-cast conv with *"Inference tensors do not track
version counter"*.

## Panel differences from the nodes

Controls are tiered exactly as the nodes tier them — `Schedule shaping`, `More
colors`, `Color shift` and `Advanced` are the node's own `advanced` / `more_colors` /
`color_shift` / `dev` booleans, gating the same branches of the maths.

Two deliberate departures from the node widgets:

- `plot_steps` is gone. It only drew tick marks on the LiteGraph plot; the webui
  already knows the step count.
- The amount sliders are **−1 to +1**, not the nodes' ±10. Measured, not guessed: a
  single application starts clipping above roughly 0.2–0.4, and a generation applies
  it at every step in its window. `saturation` keeps −1..3 because its number is a
  chroma multiplier rather than an offset. `params.py` records every range change in
  `RANGE_DEVIATIONS` and a test asserts none of them drifted by accident.

Every component sets `do_not_save_to_config = True`. That is what keeps ~60 per-image
controls out of `ui-config.json`, and it is also what makes the dynamically hidden
`mask_width` safe — the webui persists `visible` alongside `value`, so without the
opt-out a control that started hidden would stay hidden forever.

## Model support, internally

Support is keyed on the latent format's class name, which is the same on ComfyUI and
Forge:

| Latent format | Family | Vectors | Downscale | Latent |
|---|---|---|---|---|
| `Wan21` | krea2 | `colorcraft-krea2` | 8x | 16ch, 5D `[B,C,1,H,W]` |
| `Flux` | zimage | `colorcraft-zimage` | 8x | 16ch |
| `Flux2` | flux2 | `colorcraft-flux2` | 16x | 128ch, 2x2 packed |

Three things that bite when adding a family:

- **The downscale factor is per family.** It converts the mask-blur radius from image
  pixels to latent pixels. It rides in the `dev` dict from `core.resolve_dev`, because
  every consumer of a mask spec has the family and none of them has the VAE.
- **Wan-family latents are 5D.** `core`'s vector maths is 4D-only; squeeze dim 2 and
  restore, as `engine.apply_chain` does.
- **Flux2's latent is a 2x2 spatial packing** of a 32-channel space: flat channel `i`
  is unpacked channel `i // 4`, sub-pixel slot `i % 4`. A direction that differs across
  the four slots stamps a fixed 2x2 tile into every latent pixel — a 16-image-pixel
  grid. `core.project_replicated` removes that, and `build_basis_file` applies it for
  any family listed in `core.PACKED_FAMILIES`. `SDXL_Flux2` (Mugen) is the same
  channel space *unpacked*, so it needs its own vector file rather than reusing
  flux2's; that is why it is deliberately absent from the family table.

### Where the vectors come from

Krea 2's and Z-Image's are the original author's. Flux2's were derived here from the
VAE itself, using `tools/colorcraft_probe.py` — see `tools/README.md` for the
procedure. The short version: transform photographs in pixel space, encode before and
after, and take the mean latent difference; then match each axis's *visual effect*
rather than its vector norm, because equal norms are demonstrably not equal effect.

The method is validated by re-deriving the two known families and comparing against
their shipped values, which is the only reason to trust it on a family where there is
nothing to check against.

## Tests

```bash
python tests/harness.py
```

48 checks, about two seconds, no GPU or model or webui. It proves, in order: that the
shared-core refactor changed no values (against the pre-refactor `nodes.py` recovered
from git history); that the flat panel builds the same chain a node graph would; that
each family's constants reach the code that uses them; that the Forge hook's output is
**bit-identical** to the ComfyUI node's across 23 scenarios and three families; and
that the parameter table hasn't drifted from `INPUT_TYPES`.

```bash
python tests/probe_selftest.py
```

82 checks covering the derivation tool up to the point where it needs a real VAE,
including both derivation methods against a toy VAE whose answer is known
analytically.

Run them before a live generation, not after a suspicious image. Two habits worth
keeping: a comparison where both sides do nothing passes trivially, so tier 3b also
asserts the hook actually moved the latent; and no test may write into `vectors/` —
both suites redirect `core.VECTORS_DIR` to a temp copy, because `safetensors` mmaps
what it loads and on Windows a mapped file cannot be replaced.
