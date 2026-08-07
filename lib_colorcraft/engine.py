"""The per-step modifier chain — the whole of what used to be
`ColorcraftSampler.wrap`'s inner `post_cfg_function`, with its two ComfyUI
couplings lifted out into injected callables.

`ColorcraftSampler` returns a `SAMPLER`, but it is not a sampler node: the
wrapper's only job is to register a post-CFG function and then call the
untouched base solver. So the port target on Forge Neo is
`set_model_sampler_post_cfg_function`, not `p.sampler.func` — and because
Colorcraft already *is* a post-CFG effect upstream, none of the pre-CFG
linear-delta reconstruction from knowledge_skimmed_cfg.md §3 is needed.

What each frontend still has to supply:
  * `sigmas` — the full schedule. ComfyUI's sampler function receives it;
    Forge Neo stashes it at
    `model_options["transformer_options"]["sampling_sigmas"]`
    (`modules/sd_samplers_kdiffusion.py:192,:246`).
  * `cur_basis` — the resolved family's vectors, on the right device/dtype.
  * `get_anchor(color4)` — a VAE-encoded, model-space colour anchor.
"""

import torch

from . import core


# Kinds whose controls depend on the resolved basis vectors.
ADVANCED_KINDS = {"advanced", "luma", "chroma", "chroma_plus", "punch"}


def build_entries(modifiers, num_steps):
    """Resolves each chain entry's schedule against the real step count.

    Basic/Advanced own their schedule widgets inline (in `params`);
    Luma/Chroma/Punch/Shift get theirs from a separate schedule dict stashed
    on the entry."""
    built = []
    for entry in modifiers:
        p = entry["params"]
        sched = entry.get("schedule") or p
        schedule_kwargs = {k: sched[k] for k in
                           ("start", "end", "bias", "exponent", "start_off", "end_off", "smooth")}
        schedule = core.make_schedule(num_steps, amount=sched["strength"], **schedule_kwargs)
        built.append((schedule, entry["kind"], p, entry.get("mask")))
    return built


def needs_basis(built):
    """Whether anything in the chain requires the basis vectors.

    Shift works on any model and isn't in ADVANCED_KINDS, but its optional
    masking input does need a basis — so it counts only in that case."""
    return any(
        kind in ADVANCED_KINDS or (kind == "shift" and mask_params)
        for _, kind, _, mask_params in built
    )


def needed_colors(modifiers):
    """Every (red, green, blue, brightness) anchor the chain can ask for, so a
    frontend can pre-encode them instead of doing a VAE round-trip from inside
    the sampling loop. On Forge Neo that matters: a lazy `vae.encode` mid-run
    reaches `memory_management.load_models_gpu([vae.patcher])` and can evict the
    UNet.

    Takes the raw chain, not the schedule-resolved one, so it can be answered
    before the step count is known."""
    colors = set()
    for entry in modifiers:
        kind, p = entry["kind"], entry["params"]
        if "contrast" in p and p["contrast"] != 0:
            colors.add((0.0, 0.0, 0.0, 0.0))
        gated = kind == "advanced" and not p.get("color_shift")
        if kind in ("basic", "advanced", "shift") and not gated and p.get("color_shift_amount", 0) != 0:
            colors.add((p["red"], p["green"], p["blue"], p["brightness"]))
    return colors


def apply_chain(built, x0, cur_sigma, sigmas, cur_basis, family, get_anchor, work_dtype=None):
    """The per-step body. `x0` is the CFG-combined prediction (`denoised`).

    `work_dtype` casts the arithmetic — pass `torch.float32` on Forge Neo, where
    predictions arrive in fp16 and the `mean`/`amax` reductions in the vibrance
    and chroma-contrast paths run over the whole latent
    (knowledge_skimmed_cfg.md §5.2). `None` keeps the caller's dtype, which is
    what the ComfyUI node has always done.

    Returns a new tensor; never writes into `x0`. On Forge Neo the prediction
    tensors are shared with every other post-CFG hook in the list, so in-place
    edits would leak sideways (knowledge_skimmed_cfg.md §5.3)."""
    orig_dtype = x0.dtype

    is_5d = x0.dim() == 5
    if is_5d:
        x0 = x0.squeeze(2)

    if work_dtype is not None and x0.dtype != work_dtype:
        x0 = x0.to(work_dtype)
        if cur_basis is not None:
            cur_basis = {k: v.to(work_dtype) for k, v in cur_basis.items()}

    def anchor_for(color):
        return get_anchor(color).to(device=x0.device, dtype=x0.dtype)

    out = x0
    for schedule, kind, p, mask_params in built:
        s = core.sigma_to_value(cur_sigma, sigmas, schedule)
        if s == 0:
            continue

        pre = out
        if "contrast" in p and p["contrast"] != 0:
            out = core.apply_contrast(out, s * p["contrast"], anchor_for((0.0, 0.0, 0.0, 0.0)))

        if kind == "basic":
            if p["color_shift_amount"] != 0:
                out = core.apply_color_shift(
                    out, s * p["color_shift_amount"], p["mode"],
                    anchor_for((p["red"], p["green"], p["blue"], p["brightness"])),
                )

        elif kind == "advanced":
            dev = core.resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                axis1, axis2 = core.chroma_axes(cur_basis, dev["chroma_plane"])
                out = core.apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                out = core.apply_vibrance(out, axis1, axis2, s * p["vibrance"], k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                out = core.apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                out = core.apply_chroma_contrast(
                    out, axis1, axis2, s * p["chroma_contrast"],
                    r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                )
                out = core.apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])
                out = core.apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                out = core.apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
                if p["more_colors"]:
                    out = core.apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                    out = core.apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                    out = core.apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                    out = core.apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                    out = core.apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                    out = core.apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
                out = core.apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                out = core.apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])

            if p["color_shift"] and p["color_shift_amount"] != 0:
                out = core.apply_color_shift(
                    out, s * p["color_shift_amount"], p["mode"],
                    anchor_for((p["red"], p["green"], p["blue"], p["brightness"])),
                )

            # Reuses the same `dev` already resolved above for its own
            # edits to gate an external mask -- one resolution, shared.
            if mask_params and cur_basis is not None:
                out = core.apply_mask_gate(pre, out, mask_params, dev, cur_basis)

        elif kind == "luma":
            dev = core.resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = core.apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                out = core.apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])
            if mask_params and cur_basis is not None:
                out = core.apply_mask_gate(pre, out, mask_params, dev, cur_basis)

        elif kind == "chroma":
            dev = core.resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                axis1, axis2 = core.chroma_axes(cur_basis, dev["chroma_plane"])
                out = core.apply_vibrance(out, axis1, axis2, s * p["vibrance"], k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                out = core.apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                out = core.apply_chroma_contrast(
                    out, axis1, axis2, s * p["chroma_contrast"],
                    r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                )
                out = core.apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                out = core.apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
            if mask_params and cur_basis is not None:
                out = core.apply_mask_gate(pre, out, mask_params, dev, cur_basis)

        elif kind == "chroma_plus":
            dev = core.resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = core.apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                out = core.apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                out = core.apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                out = core.apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                out = core.apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                out = core.apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
            if mask_params and cur_basis is not None:
                out = core.apply_mask_gate(pre, out, mask_params, dev, cur_basis)

        elif kind == "punch":
            dev = core.resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = core.apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                out = core.apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])
            if mask_params and cur_basis is not None:
                out = core.apply_mask_gate(pre, out, mask_params, dev, cur_basis)

        elif kind == "shift":
            if p["color_shift_amount"] != 0:
                out = core.apply_color_shift(
                    out, s * p["color_shift_amount"], p["mode"],
                    anchor_for((p["red"], p["green"], p["blue"], p["brightness"])),
                )
            if mask_params and cur_basis is not None:
                dev = core.resolve_dev(p, family)
                out = core.apply_mask_gate(pre, out, mask_params, dev, cur_basis)

    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    if is_5d:
        out = out.unsqueeze(2)
    return out
