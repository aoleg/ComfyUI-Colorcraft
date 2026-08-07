import comfy.samplers
import comfy.model_patcher
import comfy.model_management

from .lib_colorcraft import core, engine
from .lib_colorcraft.core import MASK_AXIS_OPTIONS


# ---------------------------------------------------------------------------
# The vector math, schedule building and mask-tree evaluation live in
# `lib_colorcraft/core.py`, and the per-step modifier chain in
# `lib_colorcraft/engine.py`, so the Forge Neo script under `scripts/` (and, in
# time, the SwarmUI extension) run exactly the same code rather than a
# transcription of it. Everything below is the ComfyUI graph surface.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ColorcraftBasic — the basic node. No vectors, no masking; a wildcard that
# runs and should work regardless of which real model the sampler resolves to.
# ---------------------------------------------------------------------------

class ColorcraftBasic:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # -- schedule ------------------------------------------------------
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "advanced": ("BOOLEAN", {"default": False}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "smooth": ("BOOLEAN", {"default": True}),
                # UI-only -- purely for the JS schedule plot tick marks; never read
                # server-side, since the sampler already knows the real step count.
                "plot_steps": ("INT", {"default": 8, "min": 2, "max": 20, "step": 1}),

                # -- contrast --------------------------------------------------------
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # -- color_shift -------------------------------------------------------
                # No accordion here (node's small enough not to need one) and no
                # separate on/off widget -- gated on color_shift_amount != 0, the
                # same "0 = no-op" convention as everything else on this node.
                "color_shift_amount": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["default", "legacy"],),
                "red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
            },
        }

    def make(self, modifiers=None, **params):
        if not params.pop("advanced", False):
            params["exponent"] = 0.0
            params["start_off"] = 0.0
            params["end_off"] = 0.0
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "basic", "params": params})
        return (chain,)


# ---------------------------------------------------------------------------
# ColorcraftAdvanced — the mega-node. Non-basic sliders here depend
# on whatever VAE-family basis the sampler resolves against the actual
# connected VAE.
# ---------------------------------------------------------------------------

class ColorcraftAdvanced:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # -- schedule ------------------------------------------------------
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "advanced": ("BOOLEAN", {"default": False}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "smooth": ("BOOLEAN", {"default": True}),
                # UI-only -- purely for the JS schedule plot tick marks;
                "plot_steps": ("INT", {"default": 8, "min": 2, "max": 20, "step": 1}),

                # -- luma group ------------------------------------------------------
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tone_compression": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),

                # -- punch group -----------------------------------------------------
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clarity": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # -- chroma/color group -------------------------------------------------
                "temperature": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 10.0, "step": 0.01}),
                "chroma_contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "chroma_center": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                # -- chroma plus group -------------------------------------------------
                "more_colors": ("BOOLEAN", {"default": False}),
                "temp_plus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "temp_minus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_plus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_minus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # -- color_shift group (accordion) -------------------------------------
                "color_shift": ("BOOLEAN", {"default": False}),
                "color_shift_amount": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["default", "legacy"],),
                "red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),

                # -- dev group (accordion) -----------------------------------------------
                # Per-model calibrated values live in MODEL_DEV_DEFAULTS; these three
                # are overrides for a power user who wants to deviate from the calibrated
                # values, each gated by its own *_override bool.
                "dev": ("BOOLEAN", {"default": False}),
                "recenter_override": ("BOOLEAN", {"default": False}),
                "recenter": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_chroma_override": ("BOOLEAN", {"default": False}),
                "max_chroma": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "chroma_plane_override": ("BOOLEAN", {"default": False}),
                "chroma_plane": (["temp_tint", "lab"],),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, modifiers=None, masking=None, **params):
        if not params.pop("advanced", False):
            params["exponent"] = 0.0
            params["start_off"] = 0.0
            params["end_off"] = 0.0
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "advanced", "params": params, "mask": masking})
        return (chain,)


# ---------------------------------------------------------------------------
# Sub-module nodes -- the modular alternative to ColorcraftAdvanced. Any
# number chain together via `modifiers`. Each (except Masking, Schedule)
# optionally takes a `masking` input built by ColorcraftMasking (or a
# Combine/Blur tree on top of one). Luma/Chroma/Punch take a required
# `schedule` input built by ColorcraftSchedule instead of owning their own
# schedule widgets, so several can share one schedule.
# ---------------------------------------------------------------------------

class ColorcraftSchedule:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_SCHEDULE",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "advanced": ("BOOLEAN", {"default": False}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "smooth": ("BOOLEAN", {"default": True}),
                # UI-only -- purely for the JS schedule plot tick marks;
                "plot_steps": ("INT", {"default": 8, "min": 2, "max": 20, "step": 1}),
            },
        }

    def make(self, **params):
        if not params.pop("advanced", False):
            params["exponent"] = 0.0
            params["start_off"] = 0.0
            params["end_off"] = 0.0
        return (params,)


class ColorcraftLuma:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tone_compression": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "luma", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftChroma:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "temperature": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 10.0, "step": 0.01}),
                "chroma_contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "chroma_center": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "chroma", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftChromaPlus:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "temp_plus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "temp_minus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_plus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_minus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "chroma_plus", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftPunch:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clarity": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "punch", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftShift:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "color_shift_amount": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["default", "legacy"],),
                "red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "shift", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftMaskPreview:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    COLORS = {
        "red": (0.5, -0.5, -0.5),
        "green": (-0.5, 0.5, -0.5),
        "blue": (-0.5, -0.5, 0.5),
        "white": (0.5, 0.5, 0.5),
        "black": (-0.5, -0.5, -0.5),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": (list(cls.COLORS.keys()),),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, color, modifiers=None, masking=None):
        r, g, b = self.COLORS[color]
        schedule = {
            "strength": 1.0, "start": 1.0, "end": 1.0,
            "bias": 0.5, "exponent": 0.0, "start_off": 0.0, "end_off": 0.0,
            "smooth": True,
        }
        params = {
            "color_shift_amount": 1.0, "mode": "legacy",
            "red": r, "green": g, "blue": b, "brightness": 0.0,
        }
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "shift", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftMasking:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MASK",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_mode": (["highs", "lows", "split", "range", "protect range"],),
                "mask_axis": (MASK_AXIS_OPTIONS,),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_width": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mask_center": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "mask_hardness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    def make(self, **params):
        return (params,)


class ColorcraftMaskCombine:
    """Combines two COLORCRAFT_MASK specs via fuzzy set logic -- e.g. AND a
    temperature-highs mask with an exposure-highs mask to target only warm
    highlights. Chain multiple Combine nodes for 3+ masks. Just packages a
    spec; the actual math happens in resolve_mask_tensor at sampling time."""
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MASK",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("COLORCRAFT_MASK",),
                "mask_b": ("COLORCRAFT_MASK",),
                "operation": (["and", "or", "subtract", "xor"],),
            },
        }

    def make(self, mask_a, mask_b, operation):
        return ({"operation": operation, "a": mask_a, "b": mask_b},)


class ColorcraftMaskBlur:
    """Spatially blurs a COLORCRAFT_MASK's eventual influence, so it spreads
    past the exact pixels that satisfied its gate condition. A separate node
    (not folded into ColorcraftMasking) so it can wrap any point in a
    mask chain, including a Combine result. Radius is in decoded-image
    pixels, converted internally by VAE_DOWNSCALE_FACTOR. `spread` — see
    apply_mask_spread."""
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MASK",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("COLORCRAFT_MASK",),
                "radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 160.0, "step": 0.1}),
                "spread": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01}),
            },
        }

    def make(self, mask, radius, spread):
        return ({"blur": radius, "spread": spread, "a": mask},)


# ---------------------------------------------------------------------------
# ColorcraftSampler — wraps a SAMPLER. Loads every basis family (cheap, tiny
# files), then resolves once which family (if any) matches the connected
# model. Basic-node controls always work; vector-based controls get disabled
# with a console warning if nothing matches.
#
# Note this node returns a SAMPLER but does not implement one: it registers a
# post-CFG function and calls the untouched base solver. The Forge Neo port
# therefore hooks `set_model_sampler_post_cfg_function` directly and has no
# sampler wrapper at all.
# ---------------------------------------------------------------------------

class ColorcraftSampler:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "wrap"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "vae": ("VAE",),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
            },
        }

    def wrap(self, sampler, vae, modifiers=None):
        if not modifiers:
            return (sampler,)

        all_basis = core.load_all_basis()

        color_cache = {}
        basis_cache = {}
        resolved = {"checked": False, "family": None}

        def get_color_latent(color, latent_format):
            key = tuple(round(c, 4) for c in color)
            if key not in color_cache:
                device = comfy.model_management.get_torch_device()
                dims = getattr(vae, "latent_dim", None)
                if dims is None:
                    dims = getattr(latent_format, "latent_dimensions", 2)
                raw = core.build_color_latent(vae, *color, device=device)
                color_cache[key] = core.to_model_space(latent_format, raw, dims)
            return color_cache[key]

        def get_basis(family, device, dtype):
            key = (family, device, dtype)
            if key not in basis_cache:
                basis_cache[key] = {k: v.to(device=device, dtype=dtype) for k, v in all_basis[family].items()}
            return basis_cache[key]

        def resolve_family(latent_format, any_advanced):
            if resolved["checked"]:
                return resolved["family"]
            resolved["checked"] = True
            fmt_name = type(latent_format).__name__ if latent_format is not None else None
            family = core.family_for_latent_format(latent_format)
            if family is not None and family not in all_basis:
                print(f"[Colorcraft] WARNING: detected VAE family '{family}' (latent_format={fmt_name}) "
                      f"but no matching colorcraft-{family}.safetensors was found; vector-based controls "
                      f"disabled this run -- only Basic-node controls (contrast/color_shift) will work.")
                family = None
            elif family is None and any_advanced:
                print(f"[Colorcraft] WARNING: no basis matches the current model (latent_format={fmt_name}); "
                      f"vector-based controls (Advanced/Luma/Chroma/Chroma Plus/Punch/Masking) disabled this run "
                      f"-- Basic-node controls (contrast/color_shift) still work, and so does Shift's own "
                      f"color_shift effect, but any masking connected to Shift will silently have no effect.")
            resolved["family"] = family
            return family

        def wrapped_sampler_function(model, x, sigmas, *args, extra_args=None, **kwargs):
            extra_args = dict(extra_args or {})
            num_steps = len(sigmas) - 1

            built = engine.build_entries(modifiers, num_steps)
            any_advanced = engine.needs_basis(built)

            def post_cfg_function(pc_args):
                x0 = pc_args["denoised"]
                cur_sigma = pc_args["sigma"].max().item()
                latent_format = getattr(pc_args.get("model"), "latent_format", None)

                family = resolve_family(latent_format, any_advanced)
                probe = x0.squeeze(2) if x0.dim() == 5 else x0
                cur_basis = get_basis(family, probe.device, probe.dtype) if family else None

                return engine.apply_chain(
                    built, x0, cur_sigma, sigmas, cur_basis, family,
                    get_anchor=lambda color: get_color_latent(color, latent_format),
                )

            model_options = comfy.model_patcher.set_model_options_post_cfg_function(
                extra_args.get("model_options", {}), post_cfg_function,
            )
            extra_args["model_options"] = model_options

            return sampler.sampler_function(
                model, x, sigmas, *args, extra_args=extra_args, **kwargs
            )

        new_sampler = comfy.samplers.KSAMPLER(
            wrapped_sampler_function,
            extra_options=sampler.extra_options,
            inpaint_options=sampler.inpaint_options,
        )
        return (new_sampler,)


NODE_CLASS_MAPPINGS = {
    "ColorcraftBasic": ColorcraftBasic,
    "ColorcraftAdvanced": ColorcraftAdvanced,
    "ColorcraftLuma": ColorcraftLuma,
    "ColorcraftChroma": ColorcraftChroma,
    "ColorcraftChromaPlus": ColorcraftChromaPlus,
    "ColorcraftSchedule": ColorcraftSchedule,
    "ColorcraftPunch": ColorcraftPunch,
    "ColorcraftShift": ColorcraftShift,
    "ColorcraftMaskPreview": ColorcraftMaskPreview,
    "ColorcraftMasking": ColorcraftMasking,
    "ColorcraftMaskCombine": ColorcraftMaskCombine,
    "ColorcraftMaskBlur": ColorcraftMaskBlur,
    "ColorcraftSampler": ColorcraftSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorcraftBasic": "Colorcraft Basic",
    "ColorcraftAdvanced": "Colorcraft Advanced",
    "ColorcraftLuma": "Colorcraft Luma",
    "ColorcraftChroma": "Colorcraft Chroma",
    "ColorcraftChromaPlus": "Colorcraft Chroma Plus",
    "ColorcraftSchedule": "Colorcraft Schedule",
    "ColorcraftPunch": "Colorcraft Punch",
    "ColorcraftShift": "Colorcraft Shift",
    "ColorcraftMaskPreview": "Colorcraft Mask Preview",
    "ColorcraftMasking": "Colorcraft Masking",
    "ColorcraftMaskCombine": "Colorcraft Combine Masks",
    "ColorcraftMaskBlur": "Colorcraft Mask Blur",
    "ColorcraftSampler": "Colorcraft Sampler",
}
