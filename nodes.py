import os
import numpy as np
import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.model_patcher
import comfy.model_management
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Schedule building
# ---------------------------------------------------------------------------

def make_schedule(steps, start, end, bias, amount, exponent, start_off, end_off, smooth):
    """Builds a length-`steps` array of per-step values, ramping from
    `start_off` up to `amount` and back down to `end_off` between `start`
    and `end`. Snapped to the nearest step, so it can come out slightly
    asymmetric at low step counts."""
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)
    start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    if exponent >= 0:
        start_values = start_values ** exponent
    else:
        start_values = 1 - (1 - start_values) ** abs(1 / exponent)
    if start_values.any():
        start_values *= (amount - start_off)
        start_values += start_off

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    if exponent >= 0:
        end_values = end_values ** exponent
    else:
        end_values = 1 - (1 - end_values) ** abs(1 / exponent)
    if end_values.any():
        end_values *= (amount - end_off)
        end_values += end_off

    multipliers[start_idx:mid_idx + 1] = start_values
    multipliers[mid_idx:end_idx + 1] = end_values
    multipliers[:start_idx] = start_off
    multipliers[end_idx + 1:] = end_off
    return multipliers


def sigma_to_value(sigma, sigmas, schedule):
    """Maps the sigma a model eval actually ran at to a value from the discrete
    per-step `schedule` array. Adapted from Jonseed's ComfyUI port of Detail
    Daemon: https://github.com/Jonseed/ComfyUI-Detail-Daemon"""
    real_sigmas = sigmas[:-1]
    n = len(schedule)
    if n < 2 or len(real_sigmas) < 2 or sigma <= 0:
        return float(schedule[0]) if n else 0.0

    deltas = (real_sigmas - sigma).abs()
    idx = int(deltas.argmin())

    if (
        (idx == 0 and sigma >= real_sigmas[0])
        or (idx == n - 1 and sigma <= real_sigmas[-1])
        or deltas[idx] == 0
    ):
        return float(schedule[idx])

    idx_lo, idx_hi = (idx, idx - 1) if sigma > real_sigmas[idx] else (idx + 1, idx)
    sig_lo, sig_hi = real_sigmas[idx_lo], real_sigmas[idx_hi]
    if sig_hi == sig_lo:
        return float(schedule[idx_lo])
    ratio = float(max(0.0, min(1.0, (sigma - sig_lo) / (sig_hi - sig_lo))))
    return float(schedule[idx_lo] + ratio * (schedule[idx_hi] - schedule[idx_lo]))

# ---------------------------------------------------------------------------
# Mask axes with the explicit ordering for the combo lists
MASK_AXIS_OPTIONS = [
    "exposure", "hue", "saturation", "temperature", "tint",
    "temp+tint", "temp-tint", "lab-a", "lab-b", "lab-a+b", "lab-a-b",
]

# ---------------------------------------------------------------------------
# Huebert/Eigenweaver basis vectors — keyed by VAE *family*, not by any specific
# diffusion model. Krea2/QwenImage share a VAE ("krea2" bundle); Flux/Z-Image
# share another ("zimage" bundle). All available families are loaded once per
# sampler run regardless of what's actually connected -- the files are tiny.
# ---------------------------------------------------------------------------

BASIS_FAMILIES = ["krea2", "zimage"]

# Both supported VAE families downscale 8x -- used to convert
# ColorcraftMaskBlur's radius from decoded-image pixels (what the UI shows)
# to latent pixels (what gaussian_blur_mask operates on).
VAE_DOWNSCALE_FACTOR = 8

# latent_format class name -> basis family. Krea2/QwenImage report "Wan21";
# Flux/Z-Image report "Flux".
LATENT_FORMAT_TO_FAMILY = {
    "Wan21": "krea2",
    "Flux": "zimage",
}

# Per-model calibrated defaults for the chroma-plane math. `vibrance_k`/
# `exposure_scale`/`color_scale`/`hue_bias` are internal only, no UI override.
# `recenter`/`max_chroma`/`chroma_plane` are overridable on Advanced (see
# resolve_dev below) since they double as artistic controls for power users.
# `exposure_scale`/`color_scale` normalize each axis's raw projection to
# roughly the same +-1 range across models. `hue_bias` (radians) rotates
# zimage's hue angle so mask_center=0 lines up with the same visual spot on
# the gradient that krea2 was calibrated against.
MODEL_DEV_DEFAULTS = {
    "krea2":  {"vibrance_k": 1.0, "max_chroma": 2.5, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 3.5, "color_scale": 3.0, "hue_bias": 0.0},
    "zimage": {"vibrance_k": 2.0, "max_chroma": 5.0, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 7.5, "color_scale": 6.0, "hue_bias": -0.4},
}

VECTORS_DIR = os.path.join(os.path.dirname(__file__), "vectors")

def load_basis(family):
    """Loads colorcraft-<family>.safetensors. Returns a dict[name -> 1D tensor]
    or None if the file isn't present."""
    path = os.path.join(VECTORS_DIR, f"colorcraft-{family}.safetensors")
    if not os.path.isfile(path):
        return None
    return load_file(path)


# ---------------------------------------------------------------------------
# Vector-space operations (ported from Huebert/Eigenweaver)
# ---------------------------------------------------------------------------

def _flatten(x):
    """[B,C,H,W] -> ([N,C], shape-info to restore)."""
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(-1, C), (B, C, H, W)


def _unflatten(z, shape):
    B, C, H, W = shape
    return z.reshape(B, H, W, C).permute(0, 3, 1, 2)


def apply_vector_offset(x, basis, alpha):
    """Adds `alpha` to the projected coefficient along `basis` and reconstructs."""
    if alpha == 0:
        return x
    z, shape = _flatten(x)
    coeffs = z @ basis
    delta = torch.full_like(coeffs, alpha)
    z2 = z + delta.unsqueeze(1) * basis
    return _unflatten(z2, shape)


def apply_vector_scale(x, basis, alpha):
    """Scales the projected coefficient along `basis` by (1 + alpha) and
    reconstructs. Normalizes `basis` to unit length first, since this op
    uses the vector twice (project, then reconstruct) and any magnitude
    baked into the stored vector would otherwise get squared instead of
    applied once."""
    if alpha == 0:
        return x
    basis = basis / basis.norm()
    z, shape = _flatten(x)
    coeffs = z @ basis
    target = coeffs * (1.0 + alpha)
    delta = target - coeffs
    z2 = z + delta.unsqueeze(1) * basis
    out = _unflatten(z2, shape)
    return out


def apply_vibrance(x, axis1_basis, axis2_basis, alpha, k=1.0, recenter=1.0, r_max=0.0):
    """Boosts/reduces chroma along the given axis pair. At r_max=0, a
    rational-curve gain that boosts near-neutral pixels more and protects
    already-vivid ones (`k` controls how fast that protection kicks in;
    k=0 is plain linear saturation, which is what the saturation slider
    uses). At r_max!=0 (the vibrance slider), pixels near r_max (max
    chroma) are fully protected instead.

    `recenter` is a 0..1 blend: 1.0 fully corrects chroma drift but also
    removes real color casts the image had on purpose (e.g. a dusk photo's
    pink tint); lower values blend between no correction and full
    correction."""
    if alpha == 0:
        return x
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    if recenter != 0:
        orig_mean = x.mean(dim=(2, 3), keepdim=True)
    z, shape = _flatten(x)
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    r = torch.sqrt(c1 * c1 + c2 * c2)

    if r_max == 0:
        # original linear saturation behavior
        gain = alpha / (1.0 + k * r)
        delta1 = gain * c1
        delta2 = gain * c2
    else:
        # fraction of remaining chroma headroom to consume
        headroom_frac = torch.clamp(r / r_max, 0, 1)
        fade = headroom_frac ** (1 / max(k, 1e-6))
        fade = fade * fade * (3 - 2 * fade)
        protect = 1.0 - fade 

        if alpha >= 0:
            gain = (alpha * protect).clamp(0.0, 1.0)
            target_r = r + gain * (r_max - r) 

        else:
            gain = (alpha * protect).clamp(-1.0, 0.0)
            target_r = r * (1.0 + gain) 

        target_r = target_r.clamp_min(0.0)
        scale = target_r / r.clamp_min(1e-8)
        delta1 = c1 * scale - c1
        delta2 = c2 * scale - c2

    z2 = z + delta1.unsqueeze(1) * axis1_basis + delta2.unsqueeze(1) * axis2_basis
    out = _unflatten(z2, shape)

    if recenter != 0:
        new_mean = out.mean(dim=(2, 3), keepdim=True)
        out = out - recenter * (new_mean - orig_mean)
    return out


def apply_chroma_contrast(x, axis1_basis, axis2_basis, gamma, r_max=2.2, chroma_center=0.5, recenter=1.0):
    """Contrast on chroma magnitude, pivoted around a model-calibrated max
    chroma point: pixels above the pivot push further up, pixels below push
    down toward zero, each side eased independently (chroma is one-sided
    and bounded at 0, so it needs separate curves rather than one shared
    midpoint).

    `gamma` is curve steepness: 0 linear, positive steepens, negative
    flattens toward the pivot. `chroma_center` (0..1) sets the pivot as a
    fraction of `r_max`. `recenter` — see apply_vibrance."""
    if gamma == 0:
        return x
    t = -gamma
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    if recenter != 0:
        orig_mean = x.mean(dim=(2, 3), keepdim=True)
    z, shape = _flatten(x)
    B, C, H, W = shape
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    r = torch.sqrt(c1 * c1 + c2 * c2)

    p = max(chroma_center * r_max, 0.0)

    d = r - p
    pos = d.clamp_min(0.0)
    neg = (-d).clamp_min(0.0)
    pos_mx = pos.reshape(B, H * W).amax(dim=1, keepdim=True).clamp_min(1e-6).expand(-1, H * W).reshape(-1)
    neg_mx = max(p, 1e-6)  # room to fall is just the pivot itself, since r can't go below 0

    def ease(u):
        if t > 0:
            return 1 - (1 - u).pow(1.0 / (t + 1))
        return 1 - (1 - u).pow(abs(t) + 1)

    pos_y = ease((pos / pos_mx).clamp(0.0, 1.0))
    neg_y = ease((neg / neg_mx).clamp(0.0, 1.0))

    new_r = (p + pos_y * pos_mx - neg_y * neg_mx).clamp_min(0.0)
    ratio = new_r / r.clamp_min(1e-6)
    delta1 = (ratio - 1.0) * c1
    delta2 = (ratio - 1.0) * c2
    z2 = z + delta1.unsqueeze(1) * axis1_basis + delta2.unsqueeze(1) * axis2_basis
    out = _unflatten(z2, shape)
    if recenter != 0:
        new_mean = out.mean(dim=(2, 3), keepdim=True)
        out = out - recenter * (new_mean - orig_mean)
    return out


def apply_tone_compression(x, exposure_basis, ui_value):
    """UI range -1..1: 0 = no-op (backend factor 1), 1 = fully compressed to
    neutral (backend factor 0), -1 = expand tone (backend factor 2)."""
    if ui_value == 0:
        return x
    factor = 1.0 - ui_value
    alpha = factor - 1.0
    return apply_vector_scale(x, exposure_basis, alpha)


# Every axis is normalized to roughly a +-1 domain before reaching
# _mask_shape (real axes via compute_mask's `scale`, saturation via its own
# mapping, hue via dividing by pi). HARDNESS_GAIN scales the hardness widget
# up to match; `width` is used directly, so width=2 means "the whole
# domain" for every axis type, hue included.
HARDNESS_GAIN = 5.0


def _mask_shape(vals, mode, center, hardness, width, strength=1.0):
    """`strength` lerps the mask toward all-ones -- 1.0 is the mask as 
    computed, 0.0 is fully disabled. Deliberately skipped for `split`: 
    that mode's output already spans [-1, 1] and encodes a sign/direction, 
    not a [0,1] gate."""
    c, s, w = center, hardness * HARDNESS_GAIN, width
    if mode == "highs":
        mask = torch.sigmoid((vals - c) * s)
    elif mode == "lows":
        mask = torch.sigmoid(-(vals - c) * s)
    elif mode == "split":
        return torch.tanh((vals - c) * s)
    elif mode in ("range", "protect range"):
        excess = ((vals - c).abs() - w / 2.0).clamp_min(0.0)
        g = torch.exp(-0.5 * (excess * s) ** 2)
        mask = (1.0 - g) if mode == "protect range" else g
    else:
        mask = torch.ones_like(vals)
    return 1.0 + strength * (mask - 1.0)


def compute_mask(x, mask_basis, mode, center, hardness, width=0.0, strength=1.0, scale=1.0):
    """`scale` divides the raw projection before shaping, normalizing every
    real axis's range to roughly +-1 on any model. The caller picks which
    divisor applies (exposure_scale vs. color_scale)."""
    mask_basis = mask_basis / mask_basis.norm()
    z, shape = _flatten(x)
    vals = z @ mask_basis
    vals /= scale
    mask = _mask_shape(vals, mode, center, hardness, width, strength)
    B, C, H, W = shape
    return mask.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # [B,1,H,W] — broadcasts over channels


def compute_hue_mask(x, axis1_basis, axis2_basis, mode, center, hardness, width=0.0, strength=1.0, hue_bias=0.0):
    """Circular counterpart to compute_mask, for a pseudo-axis rather than a
    real stored vector: gates by hue angle (atan2 of the two chroma-plane
    projections) instead of a linear axis -- e.g. for isolating skin tones,
    which occupy an angular range rather than a linear extreme. `center` is
    in raw radians; the wrapped angular difference is normalized by pi
    before reaching _mask_shape so hardness/width feel consistent with
    every other axis. `hue_bias` (radians, per-model) rotates the raw angle
    so mask_center=0 lines up across models."""
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    z, shape = _flatten(x)
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    angle = torch.atan2(c2, c1) - hue_bias
    pi = torch.pi
    diff = (angle - center * pi + pi) % (2 * pi) - pi  # wrapped difference, range (-pi, pi]
    mask = _mask_shape(diff / pi, mode, 0.0, hardness, width, strength)
    B, C, H, W = shape
    return mask.reshape(B, H, W, 1).permute(0, 3, 1, 2)


def compute_saturation_mask(x, axis1_basis, axis2_basis, mode, center, hardness, width=0.0, r_max=2.2, strength=1.0):
    """Another pseudo-axis, for chroma magnitude r itself (always >= 0, no
    natural signed axis). Maps r=0 -> -1 and r=r_max -> +1 linearly, clamped
    beyond r_max, matching the +-1 convention every other axis uses so
    mask_center/mask_hardness stay meaningful without special units."""
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    z, shape = _flatten(x)
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    r = torch.sqrt(c1 * c1 + c2 * c2)
    if r_max == 0:
        mapped = torch.full_like(r, -1.0)
    else:
        mapped = ((r / r_max) * 2.0 - 1.0).clamp(-1.0, 1.0)
    mask = _mask_shape(mapped, mode, center, hardness, width, strength)
    B, C, H, W = shape
    return mask.reshape(B, H, W, 1).permute(0, 3, 1, 2)


def gaussian_blur_mask(mask, sigma):
    """Spatially blurs an already-resolved [B,1,H,W] mask, so its influence
    spreads past the exact pixels that satisfied its gate condition -- e.g.
    letting a fire-isolating mask also affect the glow around the flame.
    Separable (two 1D passes) with reflect padding so strength doesn't fade
    at the latent's edges. `sigma` is in latent pixels, not decoded-image
    pixels."""
    if sigma <= 0:
        return mask
    radius = max(1, int(round(sigma * 3)))
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    mask = F.pad(mask, (radius, radius, 0, 0), mode="reflect")
    mask = F.conv2d(mask, kernel.view(1, 1, 1, -1))
    mask = F.pad(mask, (0, 0, radius, radius), mode="reflect")
    mask = F.conv2d(mask, kernel.view(1, 1, -1, 1))
    return mask


def apply_mask_spread(mask, spread):
    """Compensates for the coverage a blur dilutes -- a gamma curve (like a
    compositing choke/spread on a blurred matte) so the mask's covered area
    actually grows or shrinks, rather than just washing brighter/darker.
    spread=0 is identity; positive grows the mask, negative shrinks it.
    Clamped to 0..1 first to avoid NaN from pow() on floating-point noise."""
    if spread == 0:
        return mask
    gamma = 2.0 ** (-spread)
    return torch.clamp(mask, 0.0, 1.0) ** gamma


# ---------------------------------------------------------------------------
# Whole-latent operations (contrast / color_shift) — work on any model,
# no basis vectors required.
# ---------------------------------------------------------------------------

def _color_adjust(denoised, t, anchor):
    anchor = anchor.reshape(-1)
    anchor = anchor.view((1, anchor.shape[0]) + (1,) * (denoised.dim() - 2))
    x = denoised - anchor
    reduce_dims = tuple(range(2, denoised.dim()))
    mx = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    xn = x / mx
    signs = torch.sign(xn)
    ax = xn.abs()
    if t > 0:
        y = 1 - (1 - ax).pow(1.0 / (t + 1))
    else:
        y = 1 - (1 - ax).pow(abs(t) + 1)
    return anchor + y * mx * signs


def _color_adjust_legacy(denoised, t, anchor):
    anchor = anchor.reshape(-1)
    anchor = anchor.view((1, anchor.shape[0]) + (1,) * (denoised.dim() - 2))
    return torch.lerp(denoised, anchor, t)


def apply_contrast(x, alpha, neutral_anchor):
    if alpha == 0:
        return x
    return _color_adjust(x, -alpha, neutral_anchor)


def apply_color_shift(x, alpha, mode, color_anchor):
    if alpha == 0:
        return x
    fn = _color_adjust_legacy if mode == "legacy" else _color_adjust
    return fn(x, alpha, color_anchor)


def build_color_latent(vae, red, green, blue, brightness):
    """Encodes a flat-color 512x512 image and averages over spatial (and
    temporal, if present) dims to get one anchor value per channel."""
    device = comfy.model_management.get_torch_device()
    img = torch.full((1, 512, 512, 3), 0.5, device=device)
    img[..., 0] += red
    img[..., 1] += green
    img[..., 2] += blue
    img += brightness
    latent = vae.encode(img)
    return latent.mean(dim=tuple(range(2, latent.dim())))[0]


def to_model_space(vae, latent_format, anchor):
    if latent_format is None or not hasattr(latent_format, "process_in"):
        return anchor
    dims = getattr(vae, "latent_dim", None)
    if dims is None:
        dims = getattr(latent_format, "latent_dimensions", 2)
    shaped = anchor.view((1, anchor.shape[0]) + (1,) * dims)
    return latent_format.process_in(shaped)[0].reshape(-1)


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


def resolve_dev(params, family):
    """Resolves recenter/max_chroma/chroma_plane against MODEL_DEV_DEFAULTS,
    honoring per-field *_override flags in `params` when present (only
    Advanced exposes these; other node kinds fall through to the default)."""
    dev = MODEL_DEV_DEFAULTS[family]
    return {
        "vibrance_k": dev["vibrance_k"],
        "exposure_scale": dev["exposure_scale"],
        "color_scale": dev["color_scale"],
        "hue_bias": dev["hue_bias"],
        "recenter": params["recenter"] if params.get("recenter_override") else dev["recenter"],
        "max_chroma": params["max_chroma"] if params.get("max_chroma_override") else dev["max_chroma"],
        "chroma_plane": params["chroma_plane"] if params.get("chroma_plane_override") else dev["chroma_plane"],
    }


def chroma_axes(cur_basis, chroma_plane):
    return (
        (cur_basis["lab-a"], cur_basis["lab-b"]) if chroma_plane == "lab"
        else (cur_basis["temperature"], cur_basis["tint"])
    )


def resolve_mask_tensor(mask_spec, pre, cur_basis, dev):
    """Recursively resolves a COLORCRAFT_MASK spec into a per-pixel mask
    tensor. A spec is a leaf (flat mode/axis/center/hardness/width/strength
    dict), a combine node ({"operation", "a", "b"}), or a blur node
    ({"blur", "a"}) -- combine/blur nodes resolve their children into real
    tensors first, then transform. Can only happen here, at sampling time,
    since `pre`/`cur_basis` don't exist at graph-build time.

    Operations are fuzzy (product t-norm etc.), not hard boolean, since the
    underlying gates are continuous 0..1 strengths -- which also means they
    reduce to normal boolean behavior automatically at the 0/1 extremes."""
    if "blur" in mask_spec:
        mask = resolve_mask_tensor(mask_spec["a"], pre, cur_basis, dev)
        mask = gaussian_blur_mask(mask, mask_spec["blur"] / VAE_DOWNSCALE_FACTOR)
        return apply_mask_spread(mask, mask_spec["spread"])

    if "operation" in mask_spec:
        a = resolve_mask_tensor(mask_spec["a"], pre, cur_basis, dev)
        b = resolve_mask_tensor(mask_spec["b"], pre, cur_basis, dev)
        operation = mask_spec["operation"]
        if operation == "and":
            return a * b
        if operation == "or":
            return torch.clamp(a + b - a * b, 0.0, 1.0)
        if operation == "subtract":  # a but not b
            return torch.clamp(a - b, 0.0, 1.0)
        if operation == "xor":  # in one but not both
            return torch.clamp(a + b - 2 * a * b, 0.0, 1.0)
        return a  # unreachable given the combo's fixed option list

    axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
    axis = mask_spec["mask_axis"]
    shape_args = (mask_spec["mask_mode"], mask_spec["mask_center"], mask_spec["mask_hardness"], mask_spec["mask_width"])
    if axis == "hue":
        return compute_hue_mask(pre, axis1, axis2, *shape_args, mask_spec["mask_strength"], hue_bias=dev["hue_bias"])
    if axis == "saturation":
        return compute_saturation_mask(pre, axis1, axis2, *shape_args, r_max=dev["max_chroma"], strength=mask_spec["mask_strength"])
    if axis in cur_basis:
        axis_scale = dev["exposure_scale"] if axis == "exposure" else dev["color_scale"]
        return compute_mask(pre, cur_basis[axis], *shape_args, mask_spec["mask_strength"], scale=axis_scale)
    return torch.ones_like(pre[:, :1])  # unknown axis -- full pass-through, matches the old bare "return out" no-op


def apply_mask_gate(pre, out, mask_spec, dev, cur_basis):
    """`dev` is the CONSUMING module's own resolved dev settings (from
    resolve_dev(p, family), the same dict it already built for its own
    edits) -- not derived from `mask_spec`. `mask_spec` supplies only the
    mask shape itself (or, for a combine node, the sub-specs to resolve)."""
    if cur_basis is None:
        return out
    mask = resolve_mask_tensor(mask_spec, pre, cur_basis, dev)
    return pre + mask * (out - pre)


# ---------------------------------------------------------------------------
# ColorcraftSampler — wraps a SAMPLER. Loads every basis family (cheap, tiny
# files), then resolves once which family (if any) matches the connected
# model. Basic-node controls always work; vector-based controls get disabled
# with a console warning if nothing matches.
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

        all_basis = {f: load_basis(f) for f in BASIS_FAMILIES}
        all_basis = {f: b for f, b in all_basis.items() if b is not None}

        color_cache = {}
        basis_cache = {}
        resolved = {"checked": False, "family": None}

        def get_color_latent(color, latent_format, device, dtype):
            key = tuple(round(c, 4) for c in color)
            if key not in color_cache:
                anchor = to_model_space(vae, latent_format, build_color_latent(vae, *color))
                color_cache[key] = anchor.to(device=device, dtype=dtype)
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
            family = LATENT_FORMAT_TO_FAMILY.get(fmt_name)
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

            # Kinds whose controls depend on the resolved basis vectors.
            ADVANCED_KINDS = {"advanced", "luma", "chroma", "chroma_plus", "punch"}

            built = []
            for entry in modifiers:
                p = entry["params"]
                # Basic/Advanced own their schedule widgets inline (in `p`);
                # Luma/Chroma/Punch get theirs from a required
                # COLORCRAFT_SCHEDULE input instead, stashed on the entry.
                sched = entry.get("schedule") or p
                schedule_kwargs = {k: sched[k] for k in
                                    ("start", "end", "bias", "exponent", "start_off", "end_off", "smooth")}
                schedule = make_schedule(num_steps, amount=sched["strength"], **schedule_kwargs)
                built.append((schedule, entry["kind"], p, entry.get("mask")))

            # Shift works on any model and isn't in ADVANCED_KINDS, but its
            # optional masking input does need a basis -- warn only in that case.
            any_advanced = any(
                kind in ADVANCED_KINDS or (kind == "shift" and mask_params)
                for _, kind, _, mask_params in built
            )

            def post_cfg_function(pc_args):
                x0 = pc_args["denoised"]
                cur_sigma = pc_args["sigma"].max().item()
                latent_format = getattr(pc_args.get("model"), "latent_format", None)

                is_5d = x0.dim() == 5
                if is_5d:
                    x0 = x0.squeeze(2)

                family = resolve_family(latent_format, any_advanced)
                cur_basis = get_basis(family, x0.device, x0.dtype) if family else None

                out = x0
                for i, (schedule, kind, p, mask_params) in enumerate(built):
                    s = sigma_to_value(cur_sigma, sigmas, schedule)
                    if s == 0:
                        continue

                    pre = out
                    if "contrast" in p and p["contrast"] != 0:
                        neutral_anchor = get_color_latent(
                            (0.0, 0.0, 0.0, 0.0), latent_format, out.device, out.dtype,
                        )
                        out = apply_contrast(out, s * p["contrast"], neutral_anchor)

                    if kind == "basic":
                        if p["color_shift_amount"] != 0:
                            color_anchor = get_color_latent(
                                (p["red"], p["green"], p["blue"], p["brightness"]),
                                latent_format, out.device, out.dtype,
                            )
                            out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

                    elif kind == "advanced":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                            out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                            out = apply_vibrance(out, axis1, axis2, s * p["vibrance"], k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                            out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                            out = apply_chroma_contrast(
                                out, axis1, axis2, s * p["chroma_contrast"],
                                r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                            )
                            out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])
                            out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                            out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
                            if p["more_colors"]:
                                out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                                out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                                out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                                out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                                out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                                out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
                            out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                            out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])

                        if p["color_shift"] and p["color_shift_amount"] != 0:
                            color_anchor = get_color_latent(
                                (p["red"], p["green"], p["blue"], p["brightness"]),
                                latent_format, out.device, out.dtype,
                            )
                            out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

                        # Reuses the same `dev` already resolved above for its own
                        # edits to gate an external mask -- one resolution, shared.
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis)

                    elif kind == "luma":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                            out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis)

                    elif kind == "chroma":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                            out = apply_vibrance(out, axis1, axis2, s * p["vibrance"], k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                            out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                            out = apply_chroma_contrast(
                                out, axis1, axis2, s * p["chroma_contrast"],
                                r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                            )
                            out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                            out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis)

                    elif kind == "chroma_plus":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                            out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                            out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                            out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                            out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                            out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis)

                    elif kind == "punch":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                            out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis)

                    elif kind == "shift":
                        if p["color_shift_amount"] != 0:
                            color_anchor = get_color_latent(
                                (p["red"], p["green"], p["blue"], p["brightness"]),
                                latent_format, out.device, out.dtype,
                            )
                            out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)
                        if mask_params and cur_basis is not None:
                            dev = resolve_dev(p, family)
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis)

                if is_5d:
                    out = out.unsqueeze(2)
                return out

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
