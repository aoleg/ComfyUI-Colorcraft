"""Colorcraft — vector probe (Phase 0 dev tool).

Answers one question: can the 11 shipped basis vectors be rediscovered from a
VAE alone? We can only find out on the families where the answer is already
known — Krea 2 (`Wan21`) and Z-Image (`Flux`). If a method reproduces both, we
trust it on Flux2 Klein, where there is nothing to check against.

Three candidate methods, run side by side so one session settles which (if any)
works:

  A  colour-map inversion  — pseudo-inverse of the latent format's own
     `latent_rgb_factors`. Free, no VAE. Colour axes only. Baseline.
  B  encode-differencing   — transform an image in pixel space, encode before
     and after, take the mean latent difference. Winner of the first live run:
     it beat A and C on 13 of 14 axis/model combinations. Runs at several
     deltas so a value large enough to pick up second-order terms shows up as
     a column rather than as a method that nearly works.
  C  decoder probe         — nudge each latent channel up and down, decode, and
     measure what happened to the picture. Finite differences, no autograd, so
     it cannot trip over inference-mode tensors.

     (B was predicted to return ~zero for clarity/sharpness, since sharpening
     an image does not change its average — true for a per-pixel map, and what
     the offline toy VAE shows. On both real VAEs those were among B's *best*
     axes: a real encoder has receptive fields, so local detail does move the
     channel means. The prediction was wrong; C is not the only route to the
     detail axes after all.)

Everything works in **model space** (post `process_in`), because that is the
space the sampler — and therefore Colorcraft — operates in. On Wan21 that
matters a lot: `process_in` divides by a per-channel standard deviation, so a
vector derived in raw VAE space would be wrong by a different factor per
channel.

Delete this file to remove the tool; nothing else references it.
"""

import json
import math
import os
import sys
import time

import gradio as gr
import torch


def _ensure_fresh_lib():
    """Forge's "Reload UI" re-executes everything under `scripts/` but leaves
    already-imported packages in `sys.modules`. So an edit to `lib_colorcraft`
    is invisible until a full process restart, and the scripts end up running
    against a stale core — which shows up as an AttributeError for whatever was
    just added. Drop the package when its source is newer than what's loaded.

    Idempotent: whichever script runs first re-imports and stamps, the rest see
    a matching stamp and do nothing, so both scripts share one module instance.
    Development scaffolding — safe to delete once the repo settles."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lib_colorcraft")
    if not os.path.isdir(root):
        return
    newest = max((os.path.getmtime(os.path.join(root, f))
                  for f in os.listdir(root) if f.endswith(".py")), default=0.0)
    pkg = sys.modules.get("lib_colorcraft")
    if pkg is not None and getattr(pkg, "_source_mtime", None) == newest:
        return
    for name in [n for n in list(sys.modules)
                 if n == "lib_colorcraft" or n.startswith("lib_colorcraft.")]:
        del sys.modules[name]
    import lib_colorcraft
    lib_colorcraft._source_mtime = newest


_ensure_fresh_lib()

from lib_colorcraft import core  # noqa: E402

from modules import scripts, shared
from modules.processing import logger
from modules.ui_components import InputAccordion


PRIMITIVES = ["exposure", "temperature", "tint", "lab-a", "lab-b", "clarity", "sharpness"]
COLOUR_AXES = ["exposure", "temperature", "tint", "lab-a", "lab-b"]


# ---------------------------------------------------------------------------
# Colour space
# ---------------------------------------------------------------------------

_RGB2XYZ = torch.tensor([[0.4124, 0.3576, 0.1805],
                         [0.2126, 0.7152, 0.0722],
                         [0.0193, 0.1192, 0.9505]])
_XYZ2RGB = torch.linalg.inv(_RGB2XYZ)
_WHITE = torch.tensor([0.95047, 1.00000, 1.08883])


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _einsum_ch(mat, img):
    """[3,3] applied to the channel dim of [B,3,H,W]."""
    return torch.einsum("ij,bjhw->bihw", mat.to(img), img)


def srgb_to_lab(img):
    xyz = _einsum_ch(_RGB2XYZ, srgb_to_linear(img)) / _WHITE.to(img).view(1, 3, 1, 1)
    d = 6.0 / 29.0
    f = torch.where(xyz > d ** 3, xyz.clamp_min(1e-8) ** (1.0 / 3.0), xyz / (3 * d * d) + 4.0 / 29.0)
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


def lab_to_srgb(lab):
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    d = 6.0 / 29.0
    f_inv = lambda t: torch.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))
    xyz = torch.cat([f_inv(fx), f_inv(fy), f_inv(fz)], dim=1) * _WHITE.to(lab).view(1, 3, 1, 1)
    return linear_to_srgb(_einsum_ch(_XYZ2RGB, xyz))


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    radius = max(1, int(round(sigma * 3)))
    coords = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = img.shape[1]
    x = torch.nn.functional.pad(img, (radius, radius, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = torch.nn.functional.pad(x, (0, 0, radius, radius), mode="reflect")
    return torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

def _fractal_noise(n, size, gen):
    """1/f noise — natural-ish spectrum, so it gives sharpening something to
    bite on. Flat patches would make clarity/sharpness meaningless."""
    freqs = torch.fft.fftfreq(size).abs()
    r = (freqs.view(-1, 1) ** 2 + freqs.view(1, -1) ** 2).sqrt().clamp_min(1.0 / size)
    spectrum = r ** -1.2
    out = []
    for _ in range(n):
        phase = torch.rand(3, size, size, generator=gen) * 2 * torch.pi
        amp = spectrum.unsqueeze(0) * torch.exp(1j * phase)
        img = torch.fft.ifft2(amp).real
        img = (img - img.mean()) / img.std().clamp_min(1e-6)
        out.append((img * 0.18 + 0.5).clamp(0.02, 0.98))
    return torch.stack(out)


def build_corpus(n_images, size, folder=""):
    """Returns [N,3,H,W] in 0..1 sRGB. Real photographs are better for the
    detail axes; the synthetic set exists so the probe runs with no setup."""
    gen = torch.Generator().manual_seed(20240820)

    if folder and os.path.isdir(folder):
        try:
            from PIL import Image
            import numpy as np

            def to_tensor(pil):
                return torch.from_numpy(np.asarray(pil)).float().permute(2, 0, 1) / 255.0

            paths = sorted(p for p in os.listdir(folder)
                           if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")))[:n_images]
            imgs, native = [], 0
            for name in paths:
                im = Image.open(os.path.join(folder, name)).convert("RGB")

                #   (1) the whole frame, downscaled — broad colour coverage, but
                #   LANCZOS from 12MP smooths away the fine texture the detail
                #   axes are derived from...
                s = min(im.size)
                imgs.append(to_tensor(
                    im.crop(((im.width - s) // 2, (im.height - s) // 2,
                             (im.width + s) // 2, (im.height + s) // 2))
                      .resize((size, size), Image.LANCZOS)))

                #   (2) ...so also take a crop at NATIVE resolution, which keeps
                #   real pixel-level detail at the cost of a narrow field of view.
                #   clarity/sharpness need this; the colour axes need (1).
                if im.width >= size and im.height >= size:
                    left, top = (im.width - size) // 2, (im.height - size) // 2
                    imgs.append(to_tensor(im.crop((left, top, left + size, top + size))))
                    native += 1

            if imgs:
                corpus = torch.stack(imgs)
                #   high-frequency energy, so the log says whether the corpus
                #   actually carries detail rather than us assuming it does
                lum = corpus.mean(dim=1, keepdim=True)
                hf = float(((lum - gaussian_blur(lum, 1.5)) ** 2).mean().sqrt())
                return corpus, (f"{len(paths)} files from {folder} -> {len(imgs)} images "
                                f"({len(paths)} downscaled full-frame + {native} native-res crops), "
                                f"hf-energy={hf:.4f}")
        except Exception as exc:
            logger.warning(f"[Probe] could not read images from {folder} ({exc}); using synthetic corpus")

    n_noise = max(2, n_images // 2)
    parts = [_fractal_noise(n_noise, size, gen)]

    #   colour-patch grids: broad hue/lightness coverage for the colour axes
    n_patch = max(1, (n_images - n_noise) // 2)
    for _ in range(n_patch):
        blocks = 8
        cell = torch.rand(3, blocks, blocks, generator=gen) * 0.8 + 0.1
        parts.append(torch.nn.functional.interpolate(
            cell.unsqueeze(0), size=(size, size), mode="nearest"))

    #   gradients plus hard edges, for mid/high frequency content
    for i in range(max(1, n_images - n_noise - n_patch)):
        ramp = torch.linspace(0.1, 0.9, size)
        img = torch.stack([ramp.view(-1, 1).expand(size, size),
                           ramp.view(1, -1).expand(size, size),
                           torch.full((size, size), 0.5)])
        step = 2 ** (3 + (i % 3))
        checker = ((torch.arange(size).view(-1, 1) // step + torch.arange(size).view(1, -1) // step) % 2).float()
        parts.append((img * 0.75 + checker.unsqueeze(0) * 0.25).unsqueeze(0))

    corpus = torch.cat(parts, dim=0)[:max(n_images, 4)]
    lum = corpus.mean(dim=1, keepdim=True)
    hf = float(((lum - gaussian_blur(lum, 1.5)) ** 2).mean().sqrt())
    return corpus, (f"{corpus.shape[0]} synthetic images "
                    f"(fractal noise + patches + edges), hf-energy={hf:.4f}")


# ---------------------------------------------------------------------------
# Pixel-space transforms (method B) and measurements (method C)
# ---------------------------------------------------------------------------

def transform(img, axis, delta):
    """`delta` is signed; every transform is applied symmetrically so the
    difference cancels even-order terms."""
    if axis == "exposure":
        return linear_to_srgb(srgb_to_linear(img) * (2.0 ** delta)).clamp(0, 1)
    if axis == "temperature":
        lin = srgb_to_linear(img)
        gain = torch.tensor([1.0 + delta, 1.0, 1.0 - delta]).to(img).view(1, 3, 1, 1)
        return linear_to_srgb(lin * gain).clamp(0, 1)
    if axis == "tint":
        lin = srgb_to_linear(img)
        gain = torch.tensor([1.0 - delta / 2, 1.0 + delta, 1.0 - delta / 2]).to(img).view(1, 3, 1, 1)
        return linear_to_srgb(lin * gain).clamp(0, 1)
    if axis in ("lab-a", "lab-b"):
        lab = srgb_to_lab(img)
        idx = 1 if axis == "lab-a" else 2
        lab[:, idx] = lab[:, idx] + delta * 100.0
        return lab_to_srgb(lab).clamp(0, 1)
    if axis in ("clarity", "sharpness"):
        sigma = 8.0 if axis == "clarity" else 1.5
        return (img + delta * 4.0 * (img - gaussian_blur(img, sigma))).clamp(0, 1)
    raise ValueError(axis)


def measure(img, axis):
    """A scalar per image batch. Method C finds the latent direction that most
    increases this quantity."""
    if axis == "exposure":
        lin = srgb_to_linear(img)
        return (0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]).mean(dim=(1, 2))
    if axis == "temperature":
        lin = srgb_to_linear(img)
        return (lin[:, 0] - lin[:, 2]).mean(dim=(1, 2))
    if axis == "tint":
        lin = srgb_to_linear(img)
        return (lin[:, 1] - 0.5 * (lin[:, 0] + lin[:, 2])).mean(dim=(1, 2))
    if axis in ("lab-a", "lab-b"):
        lab = srgb_to_lab(img)
        return lab[:, 1 if axis == "lab-a" else 2].mean(dim=(1, 2))
    if axis in ("clarity", "sharpness"):
        sigma = 8.0 if axis == "clarity" else 1.5
        lum = srgb_to_lab(img)[:, 0:1] / 100.0
        return ((lum - gaussian_blur(lum, sigma)) ** 2).mean(dim=(1, 2, 3))
    raise ValueError(axis)


# ---------------------------------------------------------------------------
# VAE plumbing — mirrors backend/diffusion_engine/base.py:70-89 without its
# Krea2 reference-latent bookkeeping, so the probe has no side effects on a
# later generation.
# ---------------------------------------------------------------------------

def encode_model_space(vae, img):
    """[B,3,H,W] 0..1 -> model-space latent."""
    bhwc = img.movedim(1, -1)
    if getattr(vae, "is_wan", False):
        outs = [vae.first_stage_model.process_in(vae.encode(bhwc[i:i + 1]))
                for i in range(bhwc.shape[0])]
        return torch.cat(outs, dim=0).float()
    return vae.first_stage_model.process_in(vae.encode(bhwc)).float()


def decode_model_space(vae, z):
    """model-space latent -> [B,3,H,W] 0..1. `VAE.decode` clamps, so the caller
    should keep perturbations small; the clipped fraction is reported."""
    out = vae.decode(vae.first_stage_model.process_out(z))
    if out.dim() == 5:
        out = out[:, 0]
    return out.movedim(-1, 1).float()


# ---------------------------------------------------------------------------
# The three methods
# ---------------------------------------------------------------------------

def method_a(latent_format, channels):
    """Pseudo-inverse of latent_rgb_factors: the minimum-norm latent direction
    that moves a given RGB direction. Flux2's factors describe the *unpacked*
    32 channels, so the result is replicated back across the 2x2 sub-pixels."""
    factors = getattr(latent_format, "latent_rgb_factors", None)
    if factors is None:
        return {}, "no latent_rgb_factors on this latent format"

    M = torch.tensor(factors, dtype=torch.float32)
    pinv = torch.linalg.pinv(M.T)
    rgb_axis = {
        "exposure": (1.0, 1.0, 1.0),
        "temperature": (1.0, 0.0, -1.0),
        "tint": (-0.5, 1.0, -0.5),
        "lab-a": (1.0, -1.0, 0.0),
        "lab-b": (0.5, 0.5, -1.0),
    }
    packed = M.shape[0] != channels
    out = {}
    for axis, v in rgb_axis.items():
        d = pinv @ torch.tensor(v)
        if packed:
            reps = channels // M.shape[0]
            d = d.view(-1, 1).expand(-1, reps).reshape(-1)
        out[axis] = d
    note = "ok" + (f" (replicated {M.shape[0]}->{channels})" if packed else "")
    return out, note


def method_b(vae, corpus, delta, batch):
    """Mean latent difference under a symmetric pixel-space transform.

    NOTE (corrected by the first live run): clarity/sharpness were predicted to
    come back near zero here, on the reasoning that sharpening an image does not
    change its average. That holds for a per-pixel map and is what the offline
    toy VAE shows -- but a real encoder has receptive fields, so local detail
    *does* move the channel means, and on both Krea 2 and Z-Image these were
    among B's best axes. The prediction was wrong; the method is fine."""
    out = {}
    for axis in PRIMITIVES:
        acc, n = None, 0
        for i in range(0, corpus.shape[0], batch):
            chunk = corpus[i:i + batch]
            plus = encode_model_space(vae, transform(chunk, axis, +delta))
            minus = encode_model_space(vae, transform(chunk, axis, -delta))
            diff = (plus - minus) / (2 * delta)
            reduce_dims = (0,) + tuple(range(2, diff.dim()))
            acc = diff.mean(dim=reduce_dims) if acc is None else acc + diff.mean(dim=reduce_dims)
            n += 1
        out[axis] = (acc / max(n, 1)).cpu()
    return out


def method_c(vae, corpus, n_base, eps_scale, chunk, probe_size):
    """Per-channel finite differences through the decoder: nudge channel i by
    +/-eps everywhere, decode, see how each measurement moved. The resulting
    vector of partials IS the direction that a uniform offset moves fastest
    along — which is exactly what `apply_vector_offset` does."""
    base_imgs = corpus[:max(1, n_base)]
    if base_imgs.shape[-1] != probe_size:
        base_imgs = torch.nn.functional.interpolate(
            base_imgs, size=(probe_size, probe_size), mode="area")

    z0 = encode_model_space(vae, base_imgs)
    channels = z0.shape[1]
    eps = float(z0.std().item()) * eps_scale

    totals = {axis: torch.zeros(channels) for axis in PRIMITIVES}
    clipped = 0.0
    decodes = 0

    for b in range(z0.shape[0]):
        zb = z0[b:b + 1]
        base_m = {axis: measure(decode_model_space(vae, zb), axis).item() for axis in PRIMITIVES}
        decodes += 1

        for start in range(0, channels, chunk):
            idx = list(range(start, min(start + chunk, channels)))
            batch_z = zb.repeat(2 * len(idx), *([1] * (zb.dim() - 1)))
            for j, ch in enumerate(idx):
                batch_z[2 * j, ch] += eps
                batch_z[2 * j + 1, ch] -= eps

            imgs = decode_model_space(vae, batch_z)
            decodes += imgs.shape[0]
            clipped += ((imgs <= 1e-6) | (imgs >= 1.0 - 1e-6)).float().mean().item()

            for axis in PRIMITIVES:
                m = measure(imgs, axis)
                for j, ch in enumerate(idx):
                    totals[axis][ch] += (m[2 * j] - m[2 * j + 1]).item() / (2 * eps)

        del base_m

    out = {axis: (totals[axis] / z0.shape[0]).cpu() for axis in PRIMITIVES}
    info = (f"eps={eps:.4f} ({eps_scale:g} x latent std), {z0.shape[0]} base latent(s), "
            f"{channels} channels, {decodes} decodes, "
            f"clipped~{100.0 * clipped / max(1, decodes // max(1, chunk)):.1f}% of pixels")
    return out, info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cosine(a, b):
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(torch.dot(a / na, b / nb))


def in_span_residual(v, reference):
    """How much of a derived vector falls *outside* the span of the seven
    shipped primitives.

    This separates two very different failures that a plain cosine cannot tell
    apart. A low cosine with a *low* residual means the direction is inside the
    author's basis but mixed between neighbouring axes -- which is benign and
    plausible, since those axes are far from orthogonal (krea2's
    temperature.lab-b is already 0.865). A low cosine with a *high* residual
    means we are pointing somewhere their basis does not reach, which is a real
    method failure."""
    try:
        basis = torch.stack([reference[a].float() for a in PRIMITIVES], dim=1)  # [C,7]
        sol = torch.linalg.lstsq(basis, v.unsqueeze(1)).solution
        return float((v - (basis @ sol).squeeze(1)).norm() / v.norm().clamp_min(1e-12))
    except Exception:
        return float("nan")


def replication_residual(v, channels):
    """Flux2 only: how much of the direction lives *outside* the sub-pixel
    replicated subspace. Anything non-trivial here is a 2x2 grid artifact
    waiting to happen — 16 image pixels at Flux2's downscale."""
    if channels % 4 != 0:
        return None
    w = v.view(-1, 4)
    return float((w - w.mean(dim=1, keepdim=True)).norm() / v.norm().clamp_min(1e-12))


#   Our `tint` measurement is (G - (R+B)/2); the author's axis runs the other
#   way. Measured as a consistent negative cosine on both families and all three
#   methods, so it is a convention difference, not an error. Applied only when
#   there is no reference to align against.
SIGN_CONVENTION = {"tint": -1.0}


def _probe_out_dirs():
    """`scripts.basedir()` returns the *extension* directory while scripts are
    loading, but the webui root once a callback is running -- which is why
    run_probe's output lands next to the webui, not next to this file. Check both
    so the build step finds whatever the probe actually wrote."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seen, dirs = set(), []
    for root in (scripts.basedir(), here):
        d = os.path.join(root, "probe_out")
        if d not in seen:
            seen.add(d)
            dirs.append(d)
    return dirs


def build_basis_file(method_key):
    """Phase 0.5: turn one method's raw output into a basis file the extension
    can actually load, written as colorcraft-<family>-derived.safetensors.

    Four things have to happen before derived vectors are comparable with the
    shipped ones, and all of them would otherwise confound an A/B:

      * **sub-pixel replication** (packed families only) -- Flux2's latent is a
        2x2 spatial packing, so a direction that differs between the four slots
        of a channel adds a fixed 2x2 tile to every latent pixel: a 16-pixel
        grid in the decoded image. Measured at 1-6% of the derived vectors, and
        projecting it out moves the direction by <=0.2%.
      * **sign** -- a flipped axis makes its slider run backwards.
      * **norm** -- `apply_vector_offset` does NOT normalise its basis, so |v| is
        the gain per slider unit. Derived norms are arbitrary; the shipped ones
        are calibrated. Copy theirs so the A/B compares direction, not strength.
      * **the four diagonals** -- not derived at all, but exactly the normalised
        sum/difference of their parents (cosine 1.0000 against both shipped
        files), so they are constructed here.

    With no reference (Flux2) the norms are left as derived and flagged
    uncalibrated -- picking them is Phase 3's job, not this one."""
    missing = [n for n in ("DERIVED_VARIANT", "PRIMITIVE_AXES", "DIAGONAL_AXES",
                           "PACKED_SUBPIXELS", "project_replicated", "is_packed_family")
               if not hasattr(core, n)]
    if missing:
        return (f"lib_colorcraft is stale — it is missing {', '.join(missing)}, which this "
                f"build step needs. The auto-reload guard did not catch it; restart the "
                f"webui process (Reload UI is not enough) and try again.")

    model = getattr(shared, "sd_model", None)
    vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    if vae is None:
        return "No checkpoint loaded — load the model whose basis you want to build."

    latent_format = getattr(getattr(model, "model_config", None), "latent_format", None)
    family = core.family_for_latent_format(latent_format)
    if family is None:
        fmt = type(latent_format).__name__ if latent_format is not None else None
        family = (fmt or "unknown").lower()

    safe = method_key.strip().replace("/", "-").replace("@", "_")
    src = next((path for path in
                (os.path.join(d, f"derived-{family}-method{safe}.safetensors")
                 for d in _probe_out_dirs()) if os.path.isfile(path)), None)
    if src is None:
        have = sorted({n[len(f"derived-{family}-method"):-len(".safetensors")]
                       for d in _probe_out_dirs() if os.path.isdir(d)
                       for n in os.listdir(d) if n.startswith(f"derived-{family}-method")})
        return (f"No such probe output: derived-{family}-method{safe}.safetensors — "
                f"run the probe first. Looked in: {'; '.join(_probe_out_dirs())}. "
                f"Available for '{family}': {', '.join(have) or 'none'}")

    from safetensors.torch import load_file, save_file
    derived = {k: v.float() for k, v in load_file(src).items()}
    reference = core.load_basis(family)

    lines = [f"Building colorcraft-{family}{core.DERIVED_VARIANT}.safetensors "
             f"from method {method_key} ({family})"]

    channels = next((v.shape[0] for v in derived.values()), 0)
    packed = core.is_packed_family(family, latent_format) and channels % core.PACKED_SUBPIXELS == 0

    out, flipped, rescaled, projected = {}, [], [], []
    for axis in core.PRIMITIVE_AXES:
        v = derived.get(axis)
        if v is None:
            lines.append(f"  MISSING primitive '{axis}' — this method cannot build a full basis")
            return "\n".join(lines)

        #   before sign and norm, so the norm that gets matched is the norm of
        #   the vector actually being shipped
        if packed:
            residual = replication_residual(v, channels)
            v = core.project_replicated(v)
            projected.append(f"{axis}={residual:.3f}")

        if reference is not None and axis in reference:
            ref = reference[axis].float()
            if float(torch.dot(v / v.norm(), ref / ref.norm())) < 0:
                v = -v
                flipped.append(axis)
            v = v / v.norm() * ref.norm()
            rescaled.append(f"{axis}={float(ref.norm()):.3f}")
        else:
            v = v * SIGN_CONVENTION.get(axis, 1.0)
            if axis in SIGN_CONVENTION:
                flipped.append(axis + " (convention)")
        out[axis] = v.contiguous()

    for name, (a, b, sign) in core.DIAGONAL_AXES.items():
        va, vb = out[a], out[b]
        d = va / va.norm() + sign * (vb / vb.norm())
        target = (reference[name].norm() if reference is not None and name in reference
                  else 0.5 * (va.norm() + vb.norm()))
        out[name] = (d / d.norm() * target).contiguous()

    dest = os.path.join(core.VECTORS_DIR, f"colorcraft-{family}{core.DERIVED_VARIANT}.safetensors")
    save_file(out, dest)

    lines.append(f"  source: {os.path.basename(src)}")
    lines.append(f"  {len(core.PRIMITIVE_AXES)} primitives + {len(core.DIAGONAL_AXES)} "
                 f"constructed diagonals = {len(out)} axes")
    lines.append(f"  sign-flipped: {', '.join(flipped) if flipped else 'none'}")
    if packed:
        lines.append(f"  packed family ({channels}ch = {channels // core.PACKED_SUBPIXELS} x "
                     f"{core.PACKED_SUBPIXELS} sub-pixel slots): projected onto the replicated "
                     f"subspace; fraction removed per axis: {', '.join(projected)}")
    elif channels % core.PACKED_SUBPIXELS == 0 and channels >= 64:
        lines.append(f"  NOT projected — {channels} channels divides by {core.PACKED_SUBPIXELS} but "
                     f"'{family}' is not a packed family. If this model does pack its latent, add it "
                     f"to core.PACKED_FAMILIES before shipping these vectors.")
    if reference is not None:
        lines.append(f"  norms copied from the shipped file: {', '.join(rescaled)}")
        cosines = {a: float(torch.dot(out[a] / out[a].norm(),
                                      reference[a].float() / reference[a].float().norm()))
                   for a in out if a in reference}
        worst = min(cosines, key=lambda a: cosines[a])
        lines.append(f"  cosine vs shipped after alignment: worst={worst} {cosines[worst]:+.4f}")
    else:
        lines.append("  no shipped file for this family -- norms left AS DERIVED and are "
                     "UNCALIBRATED; slider units will not match the other families")
    lines.append(f"  wrote {dest}")
    lines.append("")
    lines.append("Now turn on Colorcraft -> Advanced -> 'Use derived basis vectors' and "
                 "generate. Toggle it off for the reference image.")

    for line in lines:
        logger.info(f"[Probe] {line}")
    return "\n".join(lines)


def _offset(z, basis, alpha):
    """`core.apply_vector_offset` works on [B,C,H,W]. Wan-family latents (Krea 2,
    Qwen) carry a temporal axis and arrive as [B,C,1,H,W], so squeeze it exactly
    the way `engine.apply_chain` does before the maths and restore it after."""
    is_5d = z.dim() == 5
    x = z.squeeze(2) if is_5d else z
    out = core.apply_vector_offset(x, basis, alpha)
    return out.unsqueeze(2) if is_5d else out


def _psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else float(10.0 * torch.log10(torch.tensor(1.0 / mse)))


def compare_bases(folder, size, alphas, chunk):
    """Phase 0.5, done properly: apply the shipped and the derived basis to the
    SAME encoded image and decode both.

    The generation-based A/B turned out not to discriminate. Pushing a slider
    hard enough to see the effect drives the result into clipping — measured at
    74-89% of the blue channel pinned to zero on a temperature=0.5 run — and two
    different vectors both pinned against the same wall look identical. This
    removes sampling, scheduling and model feedback from the question and asks
    only: do these two vectors perform the same operation on a latent?

    The readout that matters is agreement relative to effect size. If
    PSNR(shipped, derived) sits well above PSNR(shipped, original), the two
    vectors are doing the same thing; if they are comparable, they are not."""
    model = getattr(shared, "sd_model", None)
    vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    if vae is None:
        return "No checkpoint loaded."

    latent_format = getattr(getattr(model, "model_config", None), "latent_format", None)
    family = core.family_for_latent_format(latent_format)
    if family is None:
        return (f"No basis family for latent_format="
                f"{type(latent_format).__name__ if latent_format is not None else None}.")

    shipped = core.load_basis(family)
    derived = core.load_basis(family, core.DERIVED_VARIANT)
    if shipped is None or derived is None:
        return (f"Need both colorcraft-{family}.safetensors and "
                f"colorcraft-{family}{core.DERIVED_VARIANT}.safetensors — "
                f"run 'Build basis file' first.")

    try:
        ladder = [float(a) for a in str(alphas).replace(",", " ").split() if a]
    except ValueError:
        ladder = [0.05, 0.1, 0.2, 0.4]
    ladder = ladder or [0.05, 0.1, 0.2, 0.4]

    lines = []

    def say(msg=""):
        lines.append(msg)
        logger.info(f"[Probe] {msg}" if msg else "[Probe]")

    corpus, note = build_corpus(2, int(size), folder)
    corpus = corpus[:2]

    with torch.inference_mode():
        z0 = encode_model_space(vae, corpus)
        base_img = decode_model_space(vae, z0)

        say("=" * 78)
        say(f"Basis comparison — {family}: shipped vs derived, same latent, no sampling")
        say(f"  corpus @{int(size)}px: {note}")
        say(f"  alphas: {', '.join(f'{a:g}' for a in ladder)}")
        say("")
        say("  effect  = PSNR vs the untouched decode (lower = stronger edit)")
        say("  agree   = PSNR shipped vs derived (higher = same operation)")
        say("  margin  = agree - effect in dB (higher = agreement dominates the edit)")
        say("  dRGB    = mean channel shift from the untouched decode")
        say("")

        recommended = {}
        for axis in PRIMITIVES:
            if axis not in shipped or axis not in derived:
                continue
            say(f"  {axis}")
            say(f"    {'alpha':>7}{'effect':>9}{'agree':>8}{'margin':>8}{'clip%':>7}"
                f"   {'dRGB shipped':<24}{'dRGB derived'}")

            batch, tags = [], []
            for a in ladder:
                for name, basis in (("shipped", shipped), ("derived", derived)):
                    v = basis[axis].to(device=z0.device, dtype=z0.dtype)
                    batch.append(_offset(z0[:1], v, a))
                    tags.append((a, name))

            decoded = []
            for i in range(0, len(batch), max(1, int(chunk))):
                decoded.append(decode_model_space(vae, torch.cat(batch[i:i + int(chunk)], dim=0)))
            decoded = torch.cat(decoded, dim=0)

            per = {}
            for (a, name), img in zip(tags, decoded):
                per[(a, name)] = img

            for a in ladder:
                s, d = per[(a, "shipped")], per[(a, "derived")]
                orig = base_img[0]
                eff = 0.5 * (_psnr(s, orig) + _psnr(d, orig))
                agree = _psnr(s, d)
                clip = 100.0 * float(((s >= 0.996) | (s <= 0.004)).float().mean())
                ds = (s - orig).mean(dim=(1, 2))
                dd = (d - orig).mean(dim=(1, 2))
                fmt = lambda t: f"[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]"
                say(f"    {a:>7g}{eff:>9.1f}{agree:>8.1f}{agree - eff:>+8.1f}{clip:>7.1f}"
                    f"   {fmt(ds):<24}{fmt(dd)}")
                if clip < 2.0 and eff < 40.0:
                    recommended[axis] = a
            say("")

        say("  usable slider values (strongest alpha still under 2% clipping):")
        for axis in PRIMITIVES:
            if axis in recommended:
                say(f"    {axis:<12} <= {recommended[axis]:g}")
            elif axis in shipped:
                say(f"    {axis:<12} every alpha tested clipped — try a smaller ladder")

    say("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3 — calibration (the seven MODEL_DEV_DEFAULTS values + vector norms)
# ---------------------------------------------------------------------------

#   Effect PSNR at alpha=0.2, measured on Z-Image with the shipped vectors. A
#   new family's norms are scaled to land here too, because matching vector
#   *norms* across families demonstrably does not match visual effect: krea 2's
#   derived exposure is ~71% as strong as the shipped one at matched norm.
EFFECT_TARGET_DB = {
    "exposure": 25.7, "temperature": 31.7, "tint": 35.6, "lab-a": 34.2,
    "lab-b": 32.0, "clarity": 35.0, "sharpness": 34.9,
}

#   The calibrated values that already exist. Printed next to the measurement so
#   a method that cannot reproduce them is caught before it is trusted on a
#   family where there is nothing to check against -- the same discipline that
#   made Phase 0 worth doing.
KNOWN_CALIBRATION = {
    "krea2":  {"exposure_scale": 3.5, "color_scale": 3.0, "max_chroma": 2.5, "hue_bias": 0.0},
    "zimage": {"exposure_scale": 7.5, "color_scale": 6.0, "max_chroma": 5.0, "hue_bias": -0.4},
}

PERCENTILES = [95.0, 98.0, 99.0, 99.5, 99.9]
DEFAULT_PERCENTILE = 99.0

COLOUR_SCALE_AXES = ["temperature", "tint", "lab-a", "lab-b"]


def _flat_pixels(z):
    """[B,C,H,W] or Wan's [B,C,1,H,W] -> [N,C], one row per latent pixel."""
    if z.dim() == 5:
        z = z.squeeze(2)
    return z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).float()


def _pct(values, q):
    """A percentile that does not care how big the tensor is (torch.quantile
    refuses inputs over ~16M elements)."""
    v = values.flatten().float()
    k = max(1, min(v.numel(), int(round(q / 100.0 * v.numel()))))
    return float(v.sort().values[k - 1])


def _hue_patches(n, size, sat=0.9, val=0.85):
    """`n` flat sRGB patches evenly spaced around the hue wheel. Flat, because
    the only thing being measured is where a pure hue lands in the latent's
    chroma plane."""
    h = torch.arange(n, dtype=torch.float32) / n            # 0..1
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = torch.full_like(h, val * (1 - sat))
    q, t = val * (1 - sat * f), val * (1 - sat * (1 - f))
    v = torch.full_like(h, val)
    table = torch.stack([
        torch.stack([v, t, p]), torch.stack([q, v, p]), torch.stack([p, v, t]),
        torch.stack([p, q, v]), torch.stack([t, p, v]), torch.stack([v, p, q]),
    ])                                                       # [6,3,n]
    rgb = table[i.long().clamp(0, 5), :, torch.arange(n)]    # [n,3]
    return rgb.view(n, 3, 1, 1).expand(n, 3, size, size).contiguous(), h * 360.0


def _unwrap(angles):
    """Removes the +-pi wrap so the hue curve is monotonic enough to interpolate
    across. Hand-rolled rather than numpy's, to keep this file torch-only."""
    out = [float(angles[0])]
    for a in angles[1:]:
        step = float(a) - out[-1]
        out.append(out[-1] + (step + torch.pi) % (2 * torch.pi) - torch.pi)
    return torch.tensor(out)


def _interp_angle(degrees, angles, at):
    """Linear interpolation on an unwrapped angle curve, used both ways: find
    the hue whose raw angle is 0, and find the raw angle of a given hue."""
    unwrapped = _unwrap(angles)
    if at is None:
        #   solve raw angle(hue) = 0, i.e. any multiple of 2pi on the unwrapped
        #   curve -- the sweep around the hue wheel covers a full turn, so where
        #   it starts decides which multiple that is.
        turn = 2 * torch.pi
        for i in range(len(unwrapped) - 1):
            a, b = float(unwrapped[i]), float(unwrapped[i + 1])
            lo, hi = min(a, b), max(a, b)
            for k in range(math.ceil(lo / turn), math.floor(hi / turn) + 1):
                frac = 0.0 if a == b else (k * turn - a) / (b - a)
                return float(degrees[i] + frac * (degrees[i + 1] - degrees[i])), None
        return None, None
    x = float(at) % 360.0
    for i in range(len(degrees) - 1):
        if float(degrees[i]) <= x <= float(degrees[i + 1]):
            span = float(degrees[i + 1] - degrees[i])
            frac = 0.0 if span == 0 else (x - float(degrees[i])) / span
            raw = float(unwrapped[i] + frac * (unwrapped[i + 1] - unwrapped[i]))
            return None, (raw + torch.pi) % (2 * torch.pi) - torch.pi
    return None, float(unwrapped[-1])


def measure_calibration(folder, size, n_images, alpha, chunk, hue_anchor, write_norms):
    """Phase 3: measure the seven per-family dev values and the norm correction,
    and print a MODEL_DEV_DEFAULTS row ready to paste into `core.py`.

    Every scale here normalises a projection onto a **unit** basis vector
    (`compute_mask` and friends divide by `basis.norm()` first), so the scales
    are a property of the latent statistics and can be measured from encodes
    alone. The norms are the opposite case: they are the per-slider-unit gain,
    they only exist in image space, and the one thing Phase 0.5 established is
    that equal norms are not equal effect -- so they are measured by decoding.

    Run it on Krea 2 and Z-Image first. The report says how far the measurement
    lands from their known values; if it cannot reproduce those, it cannot be
    trusted on Flux2 either."""
    model = getattr(shared, "sd_model", None)
    vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    if vae is None:
        return "No checkpoint loaded — load the model you want to calibrate."

    latent_format = getattr(getattr(model, "model_config", None), "latent_format", None)
    family = core.family_for_latent_format(latent_format)
    if family is None:
        fmt = type(latent_format).__name__ if latent_format is not None else None
        return (f"No basis family for latent_format={fmt}. Add it to "
                f"core.LATENT_FORMAT_TO_FAMILY first — calibration needs vectors to measure.")

    variant, basis = "", core.load_basis(family)
    if basis is None:
        variant, basis = core.DERIVED_VARIANT, core.load_basis(family, core.DERIVED_VARIANT)
    if basis is None:
        return (f"No vectors for '{family}' — run the probe and 'Build basis file' first; "
                f"calibration measures an existing basis, it does not create one.")
    basis = {k: v.float() for k, v in basis.items()}

    lines = []

    def say(msg=""):
        lines.append(msg)
        logger.info(f"[Probe] {msg}" if msg else "[Probe]")

    alpha = float(alpha) or 0.2
    known = KNOWN_CALIBRATION.get(family)

    corpus, note = build_corpus(int(n_images), int(size), folder)
    patches, hue_degrees = _hue_patches(24, min(int(size), 256))

    with torch.inference_mode():
        z = encode_model_space(vae, corpus)
        rows = _flat_pixels(z).clone()

        say("=" * 78)
        say(f"Calibration — {family}"
            f"{' (using the DERIVED vectors; no shipped file)' if variant else ''}")
        say(f"  corpus @{int(size)}px: {note}")
        say(f"  latent: shape={tuple(z.shape)} pixels={rows.shape[0]} "
            f"mean={rows.mean():.4f} std={rows.std():.4f}")
        if known:
            say(f"  this family is already calibrated — the numbers below are a TEST of the "
                f"method, not a result to adopt")
        say("")

        # -- 1. projection scales -------------------------------------------
        say("  |projection onto the unit axis|, by percentile "
            "(exposure_scale / color_scale normalise this to +-1)")
        say(f"    {'axis':<12}" + "".join(f"{q:>9g}%" for q in PERCENTILES))
        per_axis = {}
        for axis in PRIMITIVES:
            v = basis.get(axis)
            if v is None:
                continue
            p = (rows @ (v / v.norm()).to(rows)).abs()
            per_axis[axis] = {q: _pct(p, q) for q in PERCENTILES}
            say(f"    {axis:<12}" + "".join(f"{per_axis[axis][q]:>10.3f}" for q in PERCENTILES))

        colour_mean = {q: sum(per_axis[a][q] for a in COLOUR_SCALE_AXES if a in per_axis)
                          / max(1, len([a for a in COLOUR_SCALE_AXES if a in per_axis]))
                       for q in PERCENTILES}
        say(f"    {'colour mean':<12}" + "".join(f"{colour_mean[q]:>10.3f}" for q in PERCENTILES))

        # -- 2. chroma radius ------------------------------------------------
        a1 = basis["temperature"] / basis["temperature"].norm()
        a2 = basis["tint"] / basis["tint"].norm()
        c1, c2 = rows @ a1.to(rows), rows @ a2.to(rows)
        r = torch.sqrt(c1 * c1 + c2 * c2)
        chroma = {q: _pct(r, q) for q in PERCENTILES}
        say(f"    {'chroma r':<12}" + "".join(f"{chroma[q]:>10.3f}" for q in PERCENTILES)
            + "   <- max_chroma")
        say("")

        #   Which percentile is the right one is exactly what the two known
        #   families answer. On an uncalibrated family there is nothing to fit,
        #   so the default column is used and said so.
        pick = DEFAULT_PERCENTILE
        if known:
            def err(q):
                return (abs(per_axis["exposure"][q] - known["exposure_scale"]) / known["exposure_scale"]
                        + abs(colour_mean[q] - known["color_scale"]) / known["color_scale"]
                        + abs(chroma[q] - known["max_chroma"]) / known["max_chroma"])
            pick = min(PERCENTILES, key=err)
            say(f"  percentile that best reproduces this family's known values: {pick:g}% "
                f"(mean error {err(pick) / 3:.1%}); the default column is {DEFAULT_PERCENTILE:g}%")
            say(f"    exposure_scale  measured {per_axis['exposure'][pick]:.2f}  vs known {known['exposure_scale']}")
            say(f"    color_scale     measured {colour_mean[pick]:.2f}  vs known {known['color_scale']}")
            say(f"    max_chroma      measured {chroma[pick]:.2f}  vs known {known['max_chroma']}")
            say("")

        exposure_scale = per_axis["exposure"][pick]
        color_scale = colour_mean[pick]
        max_chroma = chroma[pick]

        # -- 3. hue anchor ---------------------------------------------------
        zp = _flat_pixels(encode_model_space(vae, patches)).clone()
        n_pix = zp.shape[0] // patches.shape[0]
        angles = []
        for i in range(patches.shape[0]):
            block = zp[i * n_pix:(i + 1) * n_pix]
            angles.append(float(torch.atan2((block @ a2.to(block)).mean(),
                                            (block @ a1.to(block)).mean())))
        angles = torch.tensor(angles)
        anchor_hue, _ = _interp_angle(hue_degrees, angles, None)

        say("  hue: raw chroma-plane angle of a saturated patch, by hue-wheel degree")
        say("    " + "  ".join(f"{int(d):>3}:{float(a):+.2f}"
                               for d, a in list(zip(hue_degrees, angles))[::3]))
        hue_bias = 0.0
        try:
            ref_hue = float(str(hue_anchor).strip()) if str(hue_anchor).strip() else None
        except ValueError:
            ref_hue = None
        if ref_hue is None:
            say(f"    hue where the raw angle crosses 0 on THIS model: "
                f"{'%.1f deg' % anchor_hue if anchor_hue is not None else 'not found'}")
            say("    hue_bias is relative: run this on Krea 2 (hue_bias=0 by definition), paste "
                "the degree it prints into 'Hue anchor', then re-run here.")
        else:
            _, hue_bias = _interp_angle(hue_degrees, angles, ref_hue)
            hue_bias = round(float(hue_bias), 2)
            say(f"    anchor hue {ref_hue:.1f} deg sits at raw angle {hue_bias:+.2f} rad "
                f"-> hue_bias = {hue_bias:+.2f}")
            if known:
                say(f"    known hue_bias for this family: {known['hue_bias']:+.2f}")
        say("")

        # -- 4. norms, by effect ---------------------------------------------
        say(f"  effect PSNR at alpha={alpha:g} vs the Z-Image target "
            f"(norm x 10^((measured-target)/20) lands the axis on it)")
        say(f"    {'axis':<12}{'effect':>9}{'@half':>8}{'slope':>8}{'target':>8}"
            f"{'x norm':>9}{'clip%':>7}")

        base_img = decode_model_space(vae, z[:1])
        multipliers = {}
        axes = [a for a in PRIMITIVES if a in basis]
        for i in range(0, len(axes), max(1, int(chunk))):
            group = axes[i:i + max(1, int(chunk))]
            batch = []
            for axis in group:
                v = basis[axis].to(device=z.device, dtype=z.dtype)
                batch += [_offset(z[:1], v, alpha), _offset(z[:1], v, alpha / 2)]
            decoded = decode_model_space(vae, torch.cat(batch, dim=0))
            for j, axis in enumerate(group):
                full, half = decoded[2 * j], decoded[2 * j + 1]
                eff = _psnr(full, base_img[0])
                eff_half = _psnr(half, base_img[0])
                target = EFFECT_TARGET_DB[axis]
                mult = 10.0 ** ((eff - target) / 20.0)
                multipliers[axis] = mult
                clip = 100.0 * float(((full >= 0.996) | (full <= 0.004)).float().mean())
                #   halving alpha should add ~6.02 dB. It won't if the axis is
                #   already clipping, and then the multiplier is extrapolating
                #   through a nonlinearity rather than along a line.
                say(f"    {axis:<12}{eff:>9.1f}{eff_half:>8.1f}{eff_half - eff:>+8.2f}"
                    f"{target:>8.1f}{mult:>9.3f}{clip:>7.1f}")
        say("    slope should be about +6.02 dB; further off means the measurement is "
            "clipping and the multiplier is unreliable at this alpha.")
        say("")

    # -- 5. the paste-ready row ---------------------------------------------
    vibrance_k = round(max_chroma / 2.5, 2)
    say("  " + "-" * 74)
    say(f'  MODEL_DEV_DEFAULTS["{family}"] = {{"vibrance_k": {vibrance_k}, '
        f'"max_chroma": {max_chroma:.1f}, "recenter": 0.5, "chroma_plane": "temp_tint", '
        f'"exposure_scale": {exposure_scale:.1f}, "color_scale": {color_scale:.1f}, '
        f'"hue_bias": {hue_bias:+.2f}}}')
    say("")
    say("  vibrance_k is max_chroma/2.5 — a two-point correlation between the two known "
        "families, not a law. Treat it as a starting guess and adjust by eye.")
    say("  recenter (0.5) and chroma_plane (temp_tint) are the same on both known families; "
        "nothing here measures them.")

    if write_norms:
        say("")
        say(_write_calibrated_norms(family, variant, multipliers))

    say("=" * 78)
    return "\n".join(lines)


def _write_calibrated_norms(family, variant, multipliers):
    """Applies the effect-matched multipliers and saves. Only ever writes the
    *derived* file: the shipped vectors are the reference every measurement in
    this tool is checked against, so they stay untouched."""
    from safetensors.torch import save_file

    dest = os.path.join(core.VECTORS_DIR, f"colorcraft-{family}{core.DERIVED_VARIANT}.safetensors")
    if not variant and not os.path.isfile(dest):
        return ("  norms NOT written: this family has a shipped file, which is the reference "
                "and is never overwritten. Build a derived file first if you want to rescale it.")

    source = core.load_basis(family, core.DERIVED_VARIANT)
    if source is None:
        return f"  norms NOT written: {os.path.basename(dest)} is missing."

    out = {}
    for axis in core.PRIMITIVE_AXES:
        v = source[axis].float()
        out[axis] = (v * multipliers.get(axis, 1.0)).contiguous()
    #   diagonals follow their parents, exactly as build_basis_file constructs
    #   them, so a rescale cannot leave the two halves inconsistent
    for name, (a, b, sign) in core.DIAGONAL_AXES.items():
        va, vb = out[a], out[b]
        d = va / va.norm() + sign * (vb / vb.norm())
        out[name] = (d / d.norm() * (0.5 * (va.norm() + vb.norm()))).contiguous()
    save_file(out, dest)
    return (f"  norms rescaled by effect and written to {os.path.basename(dest)}: "
            + ", ".join(f"{a} x{multipliers.get(a, 1.0):.3f}" for a in core.PRIMITIVE_AXES))


def run_probe(folder, size, n_images, use_a, use_b, use_c, deltas, enc_batch,
              n_base, eps_scale, fd_chunk, probe_size, convergence):
    t0 = time.time()
    lines = []

    def say(msg=""):
        lines.append(msg)
        logger.info(f"[Probe] {msg}" if msg else "[Probe]")

    model = getattr(shared, "sd_model", None)
    vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    if vae is None:
        #   Before the first generation `shared.sd_model` is a `FakeInitialModel`
        #   placeholder (`modules/sd_models.py:239`) with no `forge_objects` at
        #   all. Ask Forge to load whatever the checkpoint dropdown points at --
        #   the same call `processing.py:785` makes, and a no-op if the hash
        #   already matches.
        try:
            from modules import sd_models
            sd_models.forge_model_reload()
            model = getattr(shared, "sd_model", None)
            vae = getattr(getattr(model, "forge_objects", None), "vae", None)
        except Exception as exc:
            logger.warning(f"[Probe] automatic checkpoint load failed: {exc}")
    if vae is None:
        return ("No checkpoint loaded, and loading one automatically failed. "
                "Select a checkpoint at the top of the page, then run the probe again.")

    latent_format = getattr(getattr(model, "model_config", None), "latent_format", None)
    fmt_name = type(latent_format).__name__ if latent_format is not None else None
    family = core.family_for_latent_format(latent_format)

    say("=" * 72)
    say(f"Colorcraft vector probe   latent_format={fmt_name}   family={family or 'UNSUPPORTED'}")
    say(f"  vae: latent_dim={getattr(vae, 'latent_dim', '?')} "
        f"channels={getattr(vae, 'latent_channels', '?')} "
        f"is_wan={getattr(vae, 'is_wan', False)} "
        f"downscale={getattr(vae, 'downscale_ratio', '?')} dtype={getattr(vae, 'vae_dtype', '?')}")

    corpus, corpus_note = build_corpus(int(n_images), int(size), folder)
    say(f"  corpus @{int(size)}px: {corpus_note}")

    #   Everything that touches the VAE runs inside `inference_mode`, matching
    #   every in-tree call path (`backend/nn/vae.py:193`,
    #   `backend/patcher/vae.py:12`, and the decorators on
    #   encode_first_stage/decode_first_stage). Under a plain `no_grad` the
    #   weights Forge loaded in inference mode blow up inside
    #   `operations.py`'s manual-cast conv with "Inference tensors do not track
    #   version counter".
    with torch.inference_mode():
        probe_z = encode_model_space(vae, corpus[:1])
    channels = probe_z.shape[1]
    say(f"  model-space latent: shape={tuple(probe_z.shape)} channels={channels} "
        f"mean={probe_z.mean():.4f} std={probe_z.std():.4f}")

    reference = None
    if family:
        try:
            reference = core.load_basis(family)
            if reference is None:
                #   the normal state for a family being derived for the first
                #   time -- not an error, and it used to be reported as one
                #   ("object of type 'NoneType' has no len()")
                say(f"  no colorcraft-{family}.safetensors to score against — deriving "
                    f"from scratch; the table below shows norms, not cosines")
            else:
                say(f"  reference vectors: colorcraft-{family}.safetensors "
                    f"({len(reference)} axes) -- cosine scores below are the Phase 0 result")
        except Exception as exc:
            say(f"  reference vectors unavailable ({type(exc).__name__}: {exc})")
    else:
        say("  no shipped vectors for this family -- deriving only, no score")

    derived, notes = {}, {}
    with torch.inference_mode():
        if use_a:
            try:
                t = time.time()
                derived["A"], note = method_a(latent_format, channels)
                notes["A"] = note
                say(f"  method A (colour-map inversion): {note}  [{time.time() - t:.1f}s]")
            except Exception as exc:
                say(f"  method A FAILED: {type(exc).__name__}: {exc}")

        if use_b:
            #   Sweep delta rather than trusting one value: a delta large enough
            #   for the symmetric difference to pick up second-order terms would
            #   look exactly like a method that nearly works.
            try:
                values = [float(d) for d in str(deltas).replace(",", " ").split() if d]
            except ValueError:
                values = [0.15]
            for d in values or [0.15]:
                try:
                    t = time.time()
                    derived[f"B@{d:g}"] = method_b(vae, corpus, d, int(enc_batch))
                    notes[f"B@{d:g}"] = f"delta={d:g}"
                    say(f"  method B (encode-differencing) delta={d:g}"
                        f"  [{time.time() - t:.1f}s]")
                except Exception as exc:
                    say(f"  method B delta={d:g} FAILED: {type(exc).__name__}: {exc}")

            #   Half the corpus at the same delta: if the score drops, more data
            #   still helps; if it doesn't move, the gap is systematic.
            if convergence and values:
                d = values[len(values) // 2]
                half = corpus[: max(2, corpus.shape[0] // 2)]
                try:
                    derived[f"B/2@{d:g}"] = method_b(vae, half, d, int(enc_batch))
                    notes[f"B/2@{d:g}"] = f"delta={d:g}, {half.shape[0]}/{corpus.shape[0]} images"
                    say(f"  method B on half the corpus ({half.shape[0]} images) delta={d:g}")
                except Exception as exc:
                    say(f"  method B half-corpus FAILED: {type(exc).__name__}: {exc}")

        if use_c:
            try:
                t = time.time()
                derived["C"], info = method_c(vae, corpus, int(n_base), float(eps_scale),
                                              int(fd_chunk), int(probe_size))
                notes["C"] = info
                say(f"  method C (decoder probe): {info}  [{time.time() - t:.1f}s]")
            except Exception as exc:
                say(f"  method C FAILED: {type(exc).__name__}: {exc}")

    #   Those results are inference tensors; clone them out so the cosine maths
    #   and safetensors below see ordinary ones.
    derived = {k: {a: v.clone() for a, v in d.items()} for k, d in derived.items()}

    say("")
    say("-" * 72)
    if reference:
        say("cosine vs shipped vectors  (>=0.95 = method confirmed)")
    else:
        say("no reference for this family; norms only")
    say(f"  {'axis':<12}" + "".join(f"{k:>11}" for k in derived))

    scores, residuals = {}, {}
    for axis in PRIMITIVES:
        row = f"  {axis:<12}"
        for k in derived:
            v = derived[k].get(axis)
            if v is None:
                row += f"{'-':>11}"
                continue
            if reference and axis in reference:
                c = cosine(v, reference[axis].float())
                scores.setdefault(k, {})[axis] = c
                row += f"{c:>+11.4f}"
            else:
                row += f"{float(v.norm()):>11.4f}"
        say(row)

    if reference:
        say("")
        say("  |cos| means the direction matches but the sign convention differs -- harmless.")
        for k, per_axis in scores.items():
            colour = [abs(per_axis[a]) for a in COLOUR_AXES if a in per_axis]
            detail = [abs(per_axis[a]) for a in ("clarity", "sharpness") if a in per_axis]
            parts = [f"colour min|cos|={min(colour):.4f}"] if colour else []
            parts += [f"detail min|cos|={min(detail):.4f}"] if detail else ["detail n/a"]
            say(f"  method {k:<9} " + "  ".join(parts))

        #   Low cosine + low residual = right subspace, wrong mix (benign).
        #   Low cosine + high residual = pointing outside their basis (real miss).
        say("")
        say("  residual outside the span of the 7 shipped primitives"
            "  (low = correct subspace, just mixed):")
        say(f"  {'axis':<12}" + "".join(f"{k:>11}" for k in derived))
        for axis in PRIMITIVES:
            row = f"  {axis:<12}"
            for k in derived:
                v = derived[k].get(axis)
                if v is None:
                    row += f"{'-':>11}"
                    continue
                r = in_span_residual(v, reference)
                residuals.setdefault(k, {})[axis] = r
                row += f"{r:>11.4f}"
            say(row)

    say("")
    say("  norms (|v| is the per-slider-unit gain for offset axes):")
    for k in derived:
        vals = "  ".join(f"{a}={float(derived[k][a].norm()):.3f}"
                         for a in PRIMITIVES if a in derived[k])
        say(f"    {k}: {vals}")

    if channels % 4 == 0 and channels >= 64:
        say("")
        say("  sub-pixel replication residual (Flux2 packing; 0 = safe, >0 = 2x2 grid risk):")
        for k in derived:
            vals = "  ".join(f"{a}={replication_residual(derived[k][a], channels):.3f}"
                             for a in PRIMITIVES if a in derived[k])
            say(f"    {k}: {vals}")

    out_dir = _probe_out_dirs()[0]
    os.makedirs(out_dir, exist_ok=True)
    tag = family or (fmt_name or "unknown").lower()
    try:
        from safetensors.torch import save_file
        for k in derived:
            payload = {a: v.contiguous() for a, v in derived[k].items()}
            if payload:
                safe = k.replace("/", "-").replace("@", "_")  # method keys reach the filename
                save_file(payload, os.path.join(out_dir, f"derived-{tag}-method{safe}.safetensors"))
        report = {
            "latent_format": fmt_name, "family": family, "channels": channels,
            "corpus": corpus_note, "size": int(size), "notes": {k: str(v) for k, v in notes.items()},
            "cosines": scores,
            #   the most useful number in the run: a low cosine with a low
            #   residual is a remix of the author's own axes, a low cosine with a
            #   high residual is a genuine miss
            "in_span_residuals": residuals,
            "norms": {k: {a: float(derived[k][a].norm()) for a in derived[k]} for k in derived},
        }
        with open(os.path.join(out_dir, f"report-{tag}.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        say("")
        say(f"  wrote derived vectors + report to {out_dir}")
    except Exception as exc:
        say(f"  could not write output files: {exc}")

    say(f"  total {time.time() - t0:.1f}s")
    say("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------

class ColorcraftProbe(scripts.Script):
    sorting_priority = 99

    def title(self):
        return "Colorcraft — vector probe (dev)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        def keep(c):
            c.do_not_save_to_config = True
            return c

        with InputAccordion(False, label=self.title()) as acc:
            acc.do_not_save_to_config = True
            acc.accordion.do_not_save_to_config = True
            gr.HTML('<span style="opacity:.7;font-size:.85em">Phase 0: does a derivation method '
                    'reproduce the shipped vectors? Run once per checkpoint — Krea&nbsp;2 and '
                    'Z-Image are the actual test (they have reference vectors); Flux2 Klein is a '
                    'dry run. Does not generate anything. Output goes to the console and to '
                    '<code>probe_out/</code>.</span>')
            folder = keep(gr.Textbox(value="", label="Image folder (optional)",
                                     placeholder=r"e.g. T:\photos\probe  — ~20 varied photos beats the synthetic set",
                                     info="left empty, a deterministic synthetic corpus is used"))
            with gr.Row():
                size = keep(gr.Slider(256, 1024, value=512, step=64, label="Corpus resolution"))
                n_images = keep(gr.Slider(4, 48, value=12, step=1, label="Images / source files",
                                          info="with a folder, each file yields two: a downscaled "
                                               "full frame and a native-resolution crop"))
            with gr.Row():
                use_a = keep(gr.Checkbox(True, label="A · colour-map inversion"))
                use_b = keep(gr.Checkbox(True, label="B · encode-differencing"))
                use_c = keep(gr.Checkbox(True, label="C · decoder probe"))
            with gr.Accordion("Method settings", open=False):
                with gr.Row():
                    deltas = keep(gr.Textbox(value="0.05, 0.10, 0.20", label="B · transform deltas",
                                             info="comma-separated; each runs as its own column"))
                    enc_batch = keep(gr.Slider(1, 8, value=2, step=1, label="B · encode batch"))
                convergence = keep(gr.Checkbox(True, label="B · also run on half the corpus",
                                               info="if the score drops, more images still help; "
                                                    "if it holds, the remaining gap is systematic"))
                with gr.Row():
                    n_base = keep(gr.Slider(1, 6, value=2, step=1, label="C · base latents"))
                    eps_scale = keep(gr.Slider(0.02, 0.5, value=0.10, step=0.01, label="C · eps (x latent std)"))
                with gr.Row():
                    fd_chunk = keep(gr.Slider(1, 16, value=6, step=1, label="C · channels per decode batch"))
                    probe_size = keep(gr.Slider(128, 512, value=256, step=64, label="C · probe resolution"))
                gr.HTML('<span style="opacity:.7;font-size:.85em">Method C costs 2 decodes per '
                        'channel per base latent — 16 channels is quick, Flux2 has 128. Lower the '
                        'probe resolution or base count if it drags.</span>')
            run = gr.Button("Run probe", variant="primary")

            with gr.Group():
                gr.HTML('<span style="opacity:.7;font-size:.85em">Phase 0.5 — turn one '
                        'method&rsquo;s output into a loadable basis, then A/B it against the '
                        'shipped vectors via <b>Colorcraft &rarr; Advanced &rarr; Use derived '
                        'basis vectors</b>. Signs and norms are aligned to the shipped file so '
                        'the comparison is about direction, not strength.</span>')
                with gr.Row():
                    method_key = keep(gr.Textbox(value="B@0.2", label="Method to build from",
                                                 info="a column name from the run above"))
                    build = gr.Button("Build basis file")
                with gr.Row():
                    cmp_alphas = keep(gr.Textbox(value="0.05, 0.1, 0.2, 0.4",
                                                 label="Compare — slider values to sweep"))
                    cmp_chunk = keep(gr.Slider(1, 8, value=4, step=1,
                                               label="Decodes per batch"))
                compare = gr.Button("Compare bases (no sampling)")
                gr.HTML('<span style="opacity:.7;font-size:.85em">Applies both bases to the '
                        'same encoded image and decodes. Removes sampling, scheduling and model '
                        'feedback from the question, and reports where the sliders start '
                        'clipping — which the generation A/B cannot show, because two different '
                        'vectors pinned against the same clipping wall look identical.</span>')

            with gr.Group():
                gr.HTML('<span style="opacity:.7;font-size:.85em">Phase 3 &mdash; measure the '
                        'seven per-family values in <code>MODEL_DEV_DEFAULTS</code> plus the '
                        'norm correction, and print a row ready to paste into '
                        '<code>core.py</code>. Run it on <b>Krea&nbsp;2 and Z-Image first</b>: '
                        'the report says how close the measurement lands to their known values, '
                        'and a method that cannot reproduce those cannot be trusted on Flux2 '
                        'either.</span>')
                with gr.Row():
                    cal_alpha = keep(gr.Slider(0.05, 0.5, value=0.2, step=0.05,
                                               label="Slider value to measure the effect at",
                                               info="also measured at half this, to check the "
                                                    "effect is still linear there"))
                    cal_chunk = keep(gr.Slider(1, 8, value=3, step=1,
                                               label="Axes per decode batch",
                                               info="two decodes each"))
                with gr.Row():
                    hue_anchor = keep(gr.Textbox(value="", label="Hue anchor (degrees)",
                                                 placeholder="from the Krea 2 run",
                                                 info="hue_bias is relative — leave empty on "
                                                      "Krea 2, which defines it as 0"))
                    write_norms = keep(gr.Checkbox(False, label="Rescale the derived file's norms",
                                                   info="applies the measured multipliers; never "
                                                        "touches a shipped file"))
                calibrate = gr.Button("Measure calibration")

            out = keep(gr.Textbox(label="Result", lines=26, max_lines=40, show_copy_button=True))

            calibrate.click(fn=measure_calibration,
                            inputs=[folder, size, n_images, cal_alpha, cal_chunk,
                                    hue_anchor, write_norms],
                            outputs=[out])
            build.click(fn=build_basis_file, inputs=[method_key], outputs=[out])
            compare.click(fn=compare_bases,
                          inputs=[folder, size, cmp_alphas, cmp_chunk], outputs=[out])
            run.click(fn=run_probe,
                      inputs=[folder, size, n_images, use_a, use_b, use_c,
                              deltas, enc_batch, n_base, eps_scale, fd_chunk, probe_size,
                              convergence],
                      outputs=[out])

        return []
