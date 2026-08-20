"""Colorcraft for WebUI Forge Neo — Phase 1.

`ColorcraftSampler` outputs a ComfyUI `SAMPLER`, which looks like the
sampler-wrapper family (`knowledge.md` §10), but it isn't one: the wrapper
registers a post-CFG function and then calls the untouched base solver. Classify
by what the node *reads*, not by its output type — this is the same name trap
`knowledge_skimmed_cfg.md` §1 documents for "pre CFG", in a new shape.

So the hook here is `set_model_sampler_post_cfg_function`
(`backend/sampling/sampling_function.py:316-318`), registered on a UNet clone in
`process_before_every_sampling` exactly as `modules/processing_scripts/mahiro.py`
does. And because Colorcraft is *already* a post-CFG effect upstream, none of
knowledge_skimmed_cfg.md §3's linear-delta reconstruction applies — the port
reads `denoised` and returns a new `denoised`, same as the node.

Three things the ComfyUI node gets from its graph and this has to source itself:

  * **the full sigma schedule**, for `sigma_to_value`'s step mapping. Forge
    already stashes it at
    `model_options["transformer_options"]["sampling_sigmas"]`
    (`modules/sd_samplers_kdiffusion.py:192`, `:246`), written *after*
    `process_before_every_sampling` runs — so it is read lazily, on the hook's
    first invocation, never at patch time.
  * **the latent format**, which lives on `p.sd_model.model_config` here rather
    than on the UNet model, so `args["model"].latent_format` is always None.
    The class names match ComfyUI's (`Wan21`, `Flux`), so core's family table is
    shared verbatim.
  * **the colour anchors**, which need a VAE encode. Done up-front rather than
    lazily: a `vae.encode` from inside the sampling loop reaches
    `memory_management.load_models_gpu([vae.patcher])` and can evict the UNet
    mid-run.
"""

import os
import sys

import gradio as gr
import torch


def _ensure_fresh_lib():
    """Forge's "Reload UI" re-executes everything under `scripts/` but leaves
    already-imported packages in `sys.modules`, so an edit to `lib_colorcraft`
    stays invisible until a full process restart and the script runs against a
    stale core. Drop the package when its source is newer than what's loaded.

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

from lib_colorcraft import core, engine, spec  # noqa: E402
from lib_colorcraft import params as P  # noqa: E402

from modules import devices, scripts
from modules.processing import logger
from modules.ui_components import InputAccordion


#   Keyed by (first_stage_model identity, latent-format name, colour); the
#   identity half invalidates it for free on a checkpoint swap.
_ANCHOR_CACHE = {}
#   Keyed by (family, device). The vector files are a few KB, so only the
#   device/dtype move is worth caching.
_BASIS_CACHE = {}
_SHIPPED_BASIS = core.load_all_basis()


def _bundle(want=None):
    """The vector files are read once at load.

    `want` is the family this run needs, and it is what stops the cache from
    remembering an *absence*. A vectors file that appears while the webui is
    running -- dropping in a new family's file, or renaming a freshly derived
    one into place -- would otherwise stay invisible until a UI reload, and the
    symptom is silence: the log says the family has no vectors and nothing is
    graded. Re-reading only on a miss leaves the hit path untouched."""
    global _SHIPPED_BASIS
    if want is not None and want not in _SHIPPED_BASIS:
        _SHIPPED_BASIS = core.load_all_basis()
    return _SHIPPED_BASIS


def _component(param, visible=True):
    """One Gradio control from one table row.

    `do_not_save_to_config` on every single one: there are ~60 of them and they
    are per-image settings, not preferences. It also makes the dynamic
    visibility below safe — `ui_loadsave.py:91` persists `visible` alongside
    `value`/`minimum`/`maximum`/`step`, so without the opt-out a control that
    started hidden would stay hidden forever (`knowledge_sigmas.md` §5.1)."""
    kind = param["kind"]
    common = {"label": param["label"], "value": param["default"], "visible": visible}
    if param["info"]:
        common["info"] = param["info"]

    if kind == P.FLOAT:
        comp = gr.Slider(minimum=param["minimum"], maximum=param["maximum"],
                         step=param["step"], **common)
    elif kind == P.BOOL:
        comp = gr.Checkbox(**common)
    else:
        comp = gr.Dropdown(choices=param["choices"], **common)

    comp.do_not_save_to_config = True
    return comp


def _no_config(accordion):
    """`ui_loadsave.py:123-129` applies fields to both halves of an
    InputAccordion — the hidden checkbox *and* the accordion it drives — so the
    opt-out has to go on both."""
    accordion.do_not_save_to_config = True
    accordion.accordion.do_not_save_to_config = True
    return accordion


def _header(text):
    return gr.HTML(f'<span style="opacity:.6;font-size:.85em;letter-spacing:.08em;'
                   f'text-transform:uppercase">{text}</span>')


class ColorcraftNeo(scripts.Script):
    sorting_priority = 16

    #   class attributes: the hook runs long after `process_before_every_sampling`
    #   returned, and `postprocess` is not guaranteed to see the same instance
    active: bool = False
    invocations: int = 0
    diagnosed: bool = False
    warned_no_sigmas: bool = False
    warned_no_basis: bool = False

    def title(self):
        return "Colorcraft"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # -- UI ----------------------------------------------------------------

    def ui(self, is_img2img):
        c = {}

        def add(name, visible=True):
            c[name] = _component(P.ALL_BY_NAME[name], visible=visible)
            return c[name]

        def gate(name):
            param = P.ALL_BY_NAME[name]
            acc = _no_config(InputAccordion(param["default"], label=param["label"]))
            c[name] = acc
            return acc

        with _no_config(InputAccordion(False, label=self.title())) as enable:
            gr.HTML('<span style="opacity:.7;font-size:.85em">Amounts are calibrated small: '
                    'measured on Krea&nbsp;2 and Z-Image, a single application starts clipping '
                    'above ~0.2, and the schedule applies it at every step in its window. '
                    'Start around 0.1–0.3 and widen the window rather than raising the '
                    'amount.</span>')
            with gr.Group():
                _header("Schedule")
                with gr.Row():
                    add("strength")
                    add("start")
                    add("end")

            with gate("advanced"):
                with gr.Row():
                    add("bias")
                    add("smooth")
                with gr.Row():
                    add("exponent")
                    add("start_off")
                    add("end_off")

            with gr.Group():
                _header("Luma")
                with gr.Row():
                    add("exposure")
                    add("tone_compression")

            with gr.Group():
                _header("Punch")
                with gr.Row():
                    add("contrast")
                    add("clarity")
                    add("sharpness")

            with gr.Group():
                _header("Chroma")
                with gr.Row():
                    add("temperature")
                    add("tint")
                with gr.Row():
                    add("vibrance")
                    add("saturation")
                with gr.Row():
                    add("chroma_contrast")
                    add("chroma_center")

            with gate("more_colors"):
                with gr.Row():
                    add("temp_plus_tint")
                    add("temp_minus_tint")
                with gr.Row():
                    add("lab_a")
                    add("lab_b")
                with gr.Row():
                    add("lab_a_plus_b")
                    add("lab_a_minus_b")

            with gate("color_shift"):
                with gr.Row():
                    add("color_shift_amount")
                    add("mode")
                with gr.Row():
                    add("red")
                    add("green")
                    add("blue")
                    add("brightness")

            with gate("masking"):
                with gr.Group():
                    _header("Mask A")
                    with gr.Row():
                        add("mask_mode")
                        add("mask_axis")
                    with gr.Row():
                        add("mask_center")
                        add("mask_hardness")
                        add("mask_strength")
                    add("mask_width", visible=False)  # range modes only; default is "highs"

                add("mask_combine")
                with gr.Group(visible=False) as g_mask_b:
                    add("mask_operation")
                    _header("Mask B")
                    with gr.Row():
                        add("mask_b_mode")
                        add("mask_b_axis")
                    with gr.Row():
                        add("mask_b_center")
                        add("mask_b_hardness")
                        add("mask_b_strength")
                    add("mask_b_width", visible=False)

                with gr.Group():
                    _header("Blur — applied to the combined mask")
                    with gr.Row():
                        add("mask_blur_radius")
                        add("mask_spread")

                with gr.Row():
                    add("mask_preview")
                    add("mask_preview_color")

            with gate("dev"):
                gr.HTML('<span style="opacity:.6;font-size:.85em">These are calibrated '
                        'per VAE family. Each override replaces the calibrated value for '
                        'the detected model.</span>')
                with gr.Row():
                    add("recenter_override")
                    add("recenter")
                with gr.Row():
                    add("max_chroma_override")
                    add("max_chroma")
                with gr.Row():
                    add("chroma_plane_override")
                    add("chroma_plane")

            with gr.Row():
                add("apply_to_hr")
                add("debug")

        #   `width` is only read by the two range modes; everything else ignores it
        for prefix in ("", "b_"):
            c[f"mask_{prefix}mode"].change(
                fn=lambda m: gr.update(visible=m in ("range", "protect range")),
                inputs=[c[f"mask_{prefix}mode"]],
                outputs=[c[f"mask_{prefix}width"]],
                show_progress=False,
            )

        c["mask_combine"].change(
            fn=lambda on: gr.update(visible=bool(on)),
            inputs=[c["mask_combine"]],
            outputs=[g_mask_b],
            show_progress=False,
        )

        #   Nothing reaches ui-config.json, so infotext is the only persistence
        #   path. One compact key, and every field decodes from it — including
        #   back to its default when the blob is silent, so a paste fully
        #   replaces the previous look instead of leaving stale non-defaults.
        self.infotext_fields = [(enable, lambda d: spec.INFOTEXT_KEY in d)]
        self.infotext_fields += [(c[n], spec.field_getter(n)) for n in P.ordered_names()]

        return [enable] + [c[n] for n in P.ordered_names()]

    # -- plumbing ----------------------------------------------------------

    @staticmethod
    def _read(args):
        """UI args -> a complete value dict. `ui()` returns `enable` first, then
        `P.ordered_names()` in order."""
        names = P.ordered_names()
        enabled = bool(args[0]) if args else False
        values = P.defaults()
        values.update({p["name"]: p["default"] for p in P.RUNTIME_PARAMS})
        for name, value in zip(names, args[1:]):
            values[name] = value
        return enabled, values

    @classmethod
    def _reset(cls):
        cls.active = False
        cls.invocations = 0
        cls.diagnosed = False
        cls.warned_no_sigmas = False
        cls.warned_no_basis = False

    @staticmethod
    def _anchor(p, color, latent_format):
        vae = p.sd_model.forge_objects.vae
        fmt_name = type(latent_format).__name__ if latent_format is not None else "none"
        key = (id(vae.first_stage_model), fmt_name, color)
        if key not in _ANCHOR_CACHE:
            raw = core.build_color_latent(vae, *color, device=devices.device)
            dims = getattr(vae, "latent_dim", 2)
            _ANCHOR_CACHE[key] = core.to_model_space(latent_format, raw, dims)
        return _ANCHOR_CACHE[key]

    @staticmethod
    def _basis(family, device):
        """Held in fp32 so `apply_chain`'s cast is a no-op rather than a
        per-step reallocation."""
        key = (family, device)
        if key not in _BASIS_CACHE:
            _BASIS_CACHE[key] = {k: v.to(device=device, dtype=torch.float32)
                                 for k, v in _bundle(family)[family].items()}
        return _BASIS_CACHE[key]

    def before_process_batch(self, p, *args, **kwargs):
        cls = type(self)
        cls._reset()

        enabled, values = self._read(args)
        if not enabled or spec.is_no_op(values):
            return

        cls.active = True
        p.extra_generation_params[spec.INFOTEXT_KEY] = spec.to_infotext(values)

    def process_before_every_sampling(self, p, *args, **kwargs):
        cls = type(self)
        enabled, v = self._read(args)
        if not enabled or spec.is_no_op(v):
            return
        if getattr(p, "is_hr_pass", False) and not v["apply_to_hr"]:
            return

        latent_format = getattr(getattr(p.sd_model, "model_config", None), "latent_format", None)
        family = core.family_for_latent_format(latent_format)
        if family is not None and family not in _bundle(family):
            logger.warning(f"[Colorcraft] detected VAE family '{family}' but "
                           f"vectors/colorcraft-{family}.safetensors is missing; "
                           f"vector-based controls are disabled this run.")
            family = None

        chain = spec.build_chain(v)
        needs_basis = any(
            e["kind"] in engine.ADVANCED_KINDS or (e["kind"] == "shift" and e.get("mask"))
            for e in chain
        )
        if family is None and needs_basis and not cls.warned_no_basis:
            cls.warned_no_basis = True
            fmt_name = type(latent_format).__name__ if latent_format is not None else None
            logger.warning(
                f"[Colorcraft] no basis matches this model (latent_format={fmt_name}). "
                f"Support is per VAE family, not per checkpoint: Wan21 (Krea2 / Qwen-Image / "
                f"Anima / Wan) and Flux (Flux / Z-Image / Lumina2 / Chroma) are covered. "
                f"Only Contrast and Color shift will do anything this run; masking will not."
            )

        #   Pre-encode every anchor the chain can ask for, before sampling holds
        #   the UNet resident. Cached across generations by VAE identity.
        anchors = {}
        wanted = engine.needed_colors(chain)
        if wanted:
            try:
                for color in wanted:
                    anchors[color] = self._anchor(p, color, latent_format)
            except Exception as exc:
                #   Contrast and Color shift are the only anchor consumers; zero
                #   them rather than letting the hook raise mid-generation.
                anchors = {}
                for entry in chain:
                    entry["params"]["contrast"] = 0.0
                    entry["params"]["color_shift_amount"] = 0.0
                logger.warning(f"[Colorcraft] could not encode a colour anchor ({exc}); "
                               f"Contrast, Color shift and mask preview are disabled this run.")

        debug = bool(v["debug"])
        state = {"built": None, "sigmas_len": None}

        @torch.inference_mode()
        def colorcraft_post_cfg(args_):
            cls.invocations += 1
            x0 = args_["denoised"]

            sigmas = args_.get("model_options", {}).get("transformer_options", {}).get("sampling_sigmas")
            if sigmas is None:
                if not cls.warned_no_sigmas:
                    cls.warned_no_sigmas = True
                    logger.warning("[Colorcraft] no sampling_sigmas in transformer_options; "
                                   "cannot map sigma to a step, so nothing was applied.")
                return x0

            #   Built lazily: `sampling_sigmas` is written after
            #   process_before_every_sampling returns.
            if state["built"] is None or state["sigmas_len"] != len(sigmas):
                state["sigmas_len"] = len(sigmas)
                state["built"] = engine.build_entries(chain, len(sigmas) - 1)
                if debug:
                    logger.info(f"[Colorcraft] steps={len(sigmas) - 1} family={family} "
                                f"latent_format={type(latent_format).__name__ if latent_format else None} "
                                f"latent={tuple(x0.shape)}/{x0.dtype}")
                    logger.info(f"[Colorcraft] sigmas={[round(float(s), 4) for s in sigmas]}")
                    for schedule, kind, _, _ in state["built"]:
                        logger.info(f"[Colorcraft] {kind} schedule="
                                    f"{[round(float(s), 4) for s in schedule]}")

            probe = x0.squeeze(2) if x0.dim() == 5 else x0
            cur_basis = self._basis(family, probe.device) if family else None
            cur_sigma = args_["sigma"].max().item()

            if debug and cls.invocations <= state["sigmas_len"]:
                values = [round(core.sigma_to_value(cur_sigma, sigmas, s), 4)
                          for s, _, _, _ in state["built"]]
                logger.info(f"[Colorcraft] sigma={cur_sigma:.4f} -> {values}")

            return engine.apply_chain(
                state["built"], x0, cur_sigma, sigmas, cur_basis, family,
                get_anchor=lambda color: anchors[color],
                work_dtype=torch.float32,
            )

        #   `forge_objects` is reset from `forge_objects_after_applying_lora`
        #   immediately before this hook (`processing.py:1376`), so the clone
        #   never leaks into the next pass and no double-wrap guard is needed.
        unet = p.sd_model.forge_objects.unet.clone()
        unet.set_model_sampler_post_cfg_function(colorcraft_post_cfg)
        p.sd_model.forge_objects.unet = unet
        cls.active = True

    def postprocess(self, p, processed, *args):
        cls = type(self)
        if cls.active and cls.invocations == 0:
            logger.warning("[Colorcraft] enabled, but the post-CFG hook never ran — "
                           "nothing was applied to this image.")
        cls._reset()
