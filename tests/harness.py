"""Offline verification for the Forge Neo port. No GPU, no model, no webui.

Four tiers, cheapest first, following knowledge_skimmed_cfg.md §6 — with one
upgrade available here that wasn't there: this repo is under git, so the
*pre-refactor* `nodes.py` is recoverable and serves as the reference
implementation for tier 1. That turns "the shared core looks equivalent" into a
measured claim.

  1. refactor equivalence — pre-refactor `nodes.py`'s post-CFG function vs the
     current one, on the same tensors. Must be bit-identical: moving code
     between files may not change a single value.
  2. translation — `spec.build_chain` (flat UI values) vs a chain built by
     calling the real node classes. This is the tier that catches a mistranscribed
     gate or a mask assembled in the wrong order.
  3. the Forge script end to end under stubbed `modules`/`gradio` — ui() arity,
     the no-op path, hook registration, the lazy sigma read, and the hook's
     output against the ComfyUI node's on identical input.
  4. the parameter table vs `nodes.py`'s INPUT_TYPES, so the two can't drift.

Run:  python tests/harness.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stubs  # noqa: E402
from stubs import REPO  # noqa: E402

FORGE_ROOT = REPO.parent / "sd-webui-forge-classic"

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "ok  " if condition else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"   {detail}"
    print(line)
    return condition


def close(a, b, tol=0.0):
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a.double() - b.double()).abs().max().item()
    return d <= tol, f"max|diff|={d:.3e}"


# ---------------------------------------------------------------------------
# Latent formats
# ---------------------------------------------------------------------------

_lf = stubs.load_latent_formats(FORGE_ROOT)
if _lf is not None:
    FORMATS = {"krea2": _lf.Wan21(), "zimage": _lf.Flux()}
    FORMAT_SOURCE = "real (sd-webui-forge-classic)"
else:
    FORMATS = {"krea2": stubs.make_toy("Wan21"), "zimage": stubs.make_toy("Flux")}
    FORMAT_SOURCE = "toy fallback — Forge checkout not found"

LATENT_DIM = {"krea2": 3, "zimage": 2}


# ---------------------------------------------------------------------------
# Configurations exercised by every tier
# ---------------------------------------------------------------------------

def base_values():
    from lib_colorcraft import params as P
    v = P.defaults()
    v.update({p["name"]: p["default"] for p in P.RUNTIME_PARAMS})
    return v


def scenarios():
    """Each is (label, flat UI value dict). Chosen to light up every branch in
    `engine.apply_chain`, including the ones gated by a boolean."""
    out = []

    v = base_values(); v.update(exposure=0.4, contrast=0.3, start=0.0, end=1.0)
    out.append(("luma+contrast", v))

    v = base_values(); v.update(temperature=0.5, tint=-0.3, vibrance=0.6, saturation=0.4,
                                chroma_contrast=0.35, chroma_center=0.4)
    out.append(("chroma-full", v))

    v = base_values(); v.update(clarity=0.5, sharpness=-0.4, tone_compression=0.3)
    out.append(("punch+tone", v))

    v = base_values(); v.update(more_colors=True, temp_plus_tint=0.3, temp_minus_tint=-0.2,
                                lab_a=0.25, lab_b=-0.15, lab_a_plus_b=0.1, lab_a_minus_b=-0.1)
    out.append(("more-colors", v))

    v = base_values(); v.update(more_colors=False, temp_plus_tint=0.9, lab_a=0.9, exposure=0.2)
    out.append(("more-colors-gated-off", v))

    v = base_values(); v.update(color_shift=True, color_shift_amount=0.5, red=0.4,
                                green=-0.2, blue=0.3, brightness=0.1)
    out.append(("color-shift", v))

    v = base_values(); v.update(color_shift=False, color_shift_amount=0.5, red=0.4, exposure=0.2)
    out.append(("color-shift-gated-off", v))

    v = base_values(); v.update(color_shift=True, color_shift_amount=0.5, red=0.4, mode="legacy")
    out.append(("color-shift-legacy", v))

    v = base_values(); v.update(exposure=0.5, masking=True, mask_mode="highs",
                                mask_axis="exposure", mask_center=0.1, mask_hardness=2.0)
    out.append(("mask-highs-exposure", v))

    v = base_values(); v.update(temperature=0.6, masking=True, mask_mode="range",
                                mask_axis="hue", mask_center=0.3, mask_width=0.5,
                                mask_hardness=3.0, mask_strength=0.8)
    out.append(("mask-hue-range", v))

    v = base_values(); v.update(vibrance=0.5, masking=True, mask_mode="protect range",
                                mask_axis="saturation", mask_center=0.2, mask_width=0.4)
    out.append(("mask-saturation-protect", v))

    v = base_values(); v.update(exposure=0.4, masking=True, mask_mode="split",
                                mask_axis="temperature", mask_hardness=1.5)
    out.append(("mask-split-temperature", v))

    for op in ("and", "or", "subtract", "xor"):
        v = base_values(); v.update(exposure=0.45, masking=True, mask_combine=True,
                                    mask_operation=op,
                                    mask_mode="highs", mask_axis="exposure", mask_center=0.0,
                                    mask_b_mode="lows", mask_b_axis="temperature",
                                    mask_b_center=0.1, mask_b_hardness=2.0)
        out.append((f"mask-combine-{op}", v))

    v = base_values(); v.update(exposure=0.4, masking=True, mask_combine=True,
                                mask_operation="and", mask_blur_radius=24.0, mask_spread=0.5,
                                mask_axis="exposure", mask_b_axis="tint")
    out.append(("mask-combine-blur-spread", v))

    v = base_values(); v.update(masking=True, mask_preview=True, mask_preview_color="green",
                                mask_axis="exposure", mask_mode="highs")
    out.append(("mask-preview", v))

    v = base_values(); v.update(exposure=0.4, advanced=True, bias=0.3, exponent=1.5,
                                start_off=0.1, end_off=-0.1, smooth=False, start=0.1, end=0.9)
    out.append(("schedule-shaping", v))

    v = base_values(); v.update(exposure=0.4, advanced=False, exponent=1.5,
                                start_off=0.9, end_off=0.9)
    out.append(("schedule-shaping-gated-off", v))

    v = base_values(); v.update(vibrance=0.5, dev=True, recenter_override=True, recenter=0.9,
                                max_chroma_override=True, max_chroma=4.0)
    out.append(("dev-overrides", v))

    v = base_values(); v.update(temperature=0.5, vibrance=0.4, dev=True,
                                chroma_plane_override=True, chroma_plane="lab")
    out.append(("dev-lab-plane", v))

    v = base_values(); v.update(exposure=-0.5, strength=-1.0, contrast=0.4)
    out.append(("negative-strength", v))

    return out


# ---------------------------------------------------------------------------
# Chain construction via the real node classes (the tier-2 reference)
# ---------------------------------------------------------------------------

def chain_via_nodes(mod, v):
    """Builds the same chain by calling the node classes, the way a ComfyUI
    graph would. `spec.build_chain` has to agree with this."""
    mask = None
    if v["masking"]:
        (mask,) = mod.ColorcraftMasking().make(
            mask_mode=v["mask_mode"], mask_axis=v["mask_axis"],
            mask_strength=v["mask_strength"], mask_width=v["mask_width"],
            mask_center=v["mask_center"], mask_hardness=v["mask_hardness"])
        if v["mask_combine"]:
            (mask_b,) = mod.ColorcraftMasking().make(
                mask_mode=v["mask_b_mode"], mask_axis=v["mask_b_axis"],
                mask_strength=v["mask_b_strength"], mask_width=v["mask_b_width"],
                mask_center=v["mask_b_center"], mask_hardness=v["mask_b_hardness"])
            (mask,) = mod.ColorcraftMaskCombine().make(mask, mask_b, v["mask_operation"])
        if v["mask_blur_radius"] > 0 or v["mask_spread"] != 0:
            (mask,) = mod.ColorcraftMaskBlur().make(mask, v["mask_blur_radius"], v["mask_spread"])

    if v["masking"] and v["mask_preview"]:
        (chain,) = mod.ColorcraftMaskPreview().make(color=v["mask_preview_color"], masking=mask)
        return chain

    (chain,) = mod.ColorcraftAdvanced().make(
        masking=mask, plot_steps=8,
        **{k: v[k] for k in (
            "strength", "start", "end", "advanced", "bias", "exponent", "start_off",
            "end_off", "smooth", "exposure", "tone_compression", "contrast", "clarity",
            "sharpness", "temperature", "tint", "vibrance", "saturation",
            "chroma_contrast", "chroma_center", "more_colors", "temp_plus_tint",
            "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b",
            "color_shift", "color_shift_amount", "mode", "red", "green", "blue",
            "brightness", "dev", "recenter_override", "recenter", "max_chroma_override",
            "max_chroma", "chroma_plane_override", "chroma_plane")})
    return chain


def run_node_post_cfg(mod, chain, x, sigmas, latent_format, vae):
    """Drives a node module's `ColorcraftSampler` far enough to capture and call
    its post-CFG function, once per step."""
    captured = {}

    class FakeBaseSampler:
        extra_options = {}
        inpaint_options = {}

        @staticmethod
        def sampler_function(model, x_, sigmas_, *a, extra_args=None, **kw):
            fns = extra_args["model_options"]["sampler_post_cfg_function"]
            captured["fn"] = fns[-1]
            return x_

    (wrapped,) = mod.ColorcraftSampler().wrap(FakeBaseSampler, vae, chain)
    wrapped.sampler_function(None, x, sigmas, extra_args={"model_options": {}})

    fn = captured["fn"]
    model = stubs.FakeModel(latent_format)
    outs = []
    for i in range(len(sigmas) - 1):
        outs.append(fn({"denoised": x, "sigma": sigmas[i:i + 1], "model": model}))
    return outs


# ---------------------------------------------------------------------------
# Tier 1 — the refactor changed no values
# ---------------------------------------------------------------------------

def tier1_refactor_equivalence():
    print("\nTier 1 — pre-refactor nodes.py vs current, bit-identical required")

    try:
        head = subprocess.run(["git", "show", "HEAD:nodes.py"], cwd=REPO,
                              capture_output=True, check=True).stdout
    except Exception as exc:
        check("recover pre-refactor nodes.py from git", False, str(exc))
        return

    tmp = Path(tempfile.gettempdir()) / "_colorcraft_reference_nodes.py"
    tmp.write_bytes(head)
    reference = stubs.load_reference_nodes(tmp)
    current = stubs.load_current_nodes()
    check("recover pre-refactor nodes.py from git", True, f"{len(head)} bytes")

    for family, latent_format in FORMATS.items():
        vae_ref = stubs.FakeVAE(LATENT_DIM[family])
        vae_cur = stubs.FakeVAE(LATENT_DIM[family])
        for five_d in (False, True):
            sigmas = stubs.make_sigmas(6)
            x = stubs.make_latent(seed=11, five_d=five_d)
            worst = 0.0
            for label, v in scenarios():
                chain_ref = chain_via_nodes(reference, v)
                chain_cur = chain_via_nodes(current, v)
                a = run_node_post_cfg(reference, chain_ref, x, sigmas, latent_format, vae_ref)
                b = run_node_post_cfg(current, chain_cur, x, sigmas, latent_format, vae_cur)
                for ta, tb in zip(a, b):
                    ok, detail = close(ta, tb, tol=0.0)
                    if not ok:
                        check(f"{family}/{'5d' if five_d else '4d'}/{label}", False, detail)
                        break
                    worst = max(worst, (ta.double() - tb.double()).abs().max().item())
            check(f"{family} / {'5D' if five_d else '4D'} — all {len(scenarios())} scenarios",
                  True, f"max|diff|={worst:.1e} over {len(sigmas) - 1} steps each")


# ---------------------------------------------------------------------------
# Tier 2 — spec.build_chain agrees with a node-built chain
# ---------------------------------------------------------------------------

def tier2_translation():
    print("\nTier 2 — spec.build_chain vs a chain assembled from the real nodes")
    from lib_colorcraft import engine, spec

    current = stubs.load_current_nodes()

    for family, latent_format in FORMATS.items():
        basis = {k: t.to(torch.float32) for k, t in
                 __import__("lib_colorcraft", fromlist=["core"]).core.load_basis(family).items()}
        vae = stubs.FakeVAE(LATENT_DIM[family])
        sigmas = stubs.make_sigmas(8)
        x = stubs.make_latent(seed=7)
        worst, failed = 0.0, None

        for label, v in scenarios():
            node_chain = chain_via_nodes(current, v)
            spec_chain = spec.build_chain(v)

            anchors = {}
            for color in set(engine.needed_colors(node_chain)) | set(engine.needed_colors(spec_chain)):
                from lib_colorcraft import core
                raw = core.build_color_latent(vae, *color, device=torch.device("cpu"))
                anchors[color] = core.to_model_space(latent_format, raw, LATENT_DIM[family])

            outs = []
            for chain in (node_chain, spec_chain):
                built = engine.build_entries(chain, len(sigmas) - 1)
                per_step = [engine.apply_chain(built, x, float(sigmas[i]), sigmas, basis, family,
                                               get_anchor=lambda c: anchors[c])
                            for i in range(len(sigmas) - 1)]
                outs.append(per_step)

            for ta, tb in zip(*outs):
                d = (ta.double() - tb.double()).abs().max().item()
                worst = max(worst, d)
                if d > 0.0:
                    failed = f"{label}: max|diff|={d:.3e}"
                    break
            if failed:
                break

        check(f"{family} — {len(scenarios())} scenarios translate identically",
              failed is None, failed or f"max|diff|={worst:.1e}")


# ---------------------------------------------------------------------------
# Tier 3 — the Forge script
# ---------------------------------------------------------------------------

def load_forge_script():
    stubs.install_modules()
    stubs.install_gradio()
    import importlib
    if "colorcraft_neo" in sys.modules:
        return sys.modules["colorcraft_neo"]
    sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("colorcraft_neo")


def ui_args(script_mod, values, enabled=True):
    from lib_colorcraft import params as P
    return [enabled] + [values[n] for n in P.ordered_names()]


def tier3_forge_script():
    print("\nTier 3 — the Forge Neo script under stubbed modules/gradio")
    from lib_colorcraft import params as P, spec

    mod = load_forge_script()
    script = mod.ColorcraftNeo()

    returned = script.ui(False)
    check("ui() returns enable + every table param, in order",
          len(returned) == 1 + len(P.ordered_names()),
          f"{len(returned)} components")

    missing = [c for c in returned if not getattr(c, "do_not_save_to_config", False)]
    check("every component opts out of ui-config.json", not missing,
          f"{len(missing)} missing" if missing else "")

    check("infotext binds one key plus a decoder per field",
          len(script.infotext_fields) == 1 + len(P.ordered_names()))

    #   disabled -> no patch at all
    vae = stubs.FakeVAE(LATENT_DIM["krea2"])
    p = stubs.FakeP(FORMATS["krea2"], vae)
    v = base_values(); v.update(exposure=0.5)
    script.before_process_batch(p, *ui_args(mod, v, enabled=False))
    script.process_before_every_sampling(p, *ui_args(mod, v, enabled=False))
    check("disabled: no post-CFG function registered",
          len(p.sd_model.forge_objects.unet.post_cfg) == 0)

    #   enabled but every amount at 0 -> also no patch
    p = stubs.FakeP(FORMATS["krea2"], vae)
    script.before_process_batch(p, *ui_args(mod, base_values()))
    script.process_before_every_sampling(p, *ui_args(mod, base_values()))
    check("no-op settings: no post-CFG function registered",
          len(p.sd_model.forge_objects.unet.post_cfg) == 0)
    check("no-op settings: no infotext written",
          spec.INFOTEXT_KEY not in p.sd_model.forge_objects.unet.model_options.get("x", {})
          and spec.INFOTEXT_KEY not in p.extra_generation_params)

    #   hires opt-out
    p = stubs.FakeP(FORMATS["krea2"], vae, is_hr_pass=True)
    v = base_values(); v.update(exposure=0.5, apply_to_hr=False)
    script.before_process_batch(p, *ui_args(mod, v))
    script.process_before_every_sampling(p, *ui_args(mod, v))
    check("apply_to_hr=False skips the hires pass",
          len(p.sd_model.forge_objects.unet.post_cfg) == 0)

    p = stubs.FakeP(FORMATS["krea2"], vae, is_hr_pass=True)
    v = base_values(); v.update(exposure=0.5, apply_to_hr=True)
    script.before_process_batch(p, *ui_args(mod, v))
    script.process_before_every_sampling(p, *ui_args(mod, v))
    check("apply_to_hr=True patches the hires pass",
          len(p.sd_model.forge_objects.unet.post_cfg) == 1)

    #   missing sigmas -> pass-through plus one warning, never a crash
    stubs.LOGGER.lines.clear()
    p = stubs.FakeP(FORMATS["krea2"], vae)
    v = base_values(); v.update(exposure=0.5)
    script.before_process_batch(p, *ui_args(mod, v))
    script.process_before_every_sampling(p, *ui_args(mod, v))
    fn = p.sd_model.forge_objects.unet.post_cfg[-1]
    x = stubs.make_latent(seed=3, five_d=True)
    out = fn({"denoised": x, "sigma": torch.tensor([0.5]), "model_options": {}})
    check("missing sampling_sigmas: returns input untouched",
          out is x and any("sampling_sigmas" in t for t in stubs.LOGGER.texts()))

    #   unsupported model family -> a specific warning, Basic axes still run
    stubs.LOGGER.lines.clear()
    sdxl = stubs.make_toy("SDXL")
    p = stubs.FakeP(sdxl, stubs.FakeVAE(2))
    #   start/end span the whole run so the assertion below is about the missing
    #   basis, not about which step the default schedule happens to cover
    v = base_values(); v.update(exposure=0.5, contrast=0.4, start=0.0, end=1.0)
    script.before_process_batch(p, *ui_args(mod, v))
    script.process_before_every_sampling(p, *ui_args(mod, v))
    warned = any("no basis matches" in t for t in stubs.LOGGER.texts())
    check("unsupported latent format: warns, still patches for Contrast", warned
          and len(p.sd_model.forge_objects.unet.post_cfg) == 1)

    sigmas = stubs.make_sigmas(8)
    fn = p.sd_model.forge_objects.unet.post_cfg[-1]
    x = stubs.make_latent(seed=5)
    step_args = {"model_options": {"transformer_options": {"sampling_sigmas": sigmas}}}
    out = fn({"denoised": x, "sigma": sigmas[3:4], **step_args})
    check("unsupported latent format: Contrast still changes the latent",
          not torch.equal(out, x))

    #   exposure needs the basis and must have been skipped, so the result has to
    #   equal contrast acting alone
    from lib_colorcraft import core
    from lib_colorcraft import engine as _engine
    only_contrast = base_values(); only_contrast.update(contrast=0.4, start=0.0, end=1.0)
    ref_chain = spec.build_chain(only_contrast)
    ref_built = _engine.build_entries(ref_chain, len(sigmas) - 1)
    ref_anchor = core.to_model_space(
        sdxl, core.build_color_latent(p.sd_model.forge_objects.vae, 0.0, 0.0, 0.0, 0.0,
                                      device=torch.device("cpu")), 2)
    ref = _engine.apply_chain(ref_built, x, float(sigmas[3]), sigmas, None, None,
                              get_anchor=lambda c: ref_anchor, work_dtype=torch.float32)
    ok, detail = close(out, ref, tol=0.0)
    check("unsupported latent format: vector axes are skipped, not misapplied", ok, detail)

    #   a step outside the schedule window is an exact pass-through
    v_win = base_values(); v_win.update(exposure=0.5, contrast=0.4, start=0.5, end=0.75)
    p2 = stubs.FakeP(FORMATS["krea2"], stubs.FakeVAE(LATENT_DIM["krea2"]))
    script.before_process_batch(p2, *ui_args(mod, v_win))
    script.process_before_every_sampling(p2, *ui_args(mod, v_win))
    fn2 = p2.sd_model.forge_objects.unet.post_cfg[-1]
    x5 = stubs.make_latent(seed=9, five_d=True)
    outside = fn2({"denoised": x5, "sigma": sigmas[0:1], **step_args})
    inside = fn2({"denoised": x5, "sigma": sigmas[5:6], **step_args})
    check("a step outside the schedule window is an exact pass-through",
          torch.equal(outside, x5) and not torch.equal(inside, x5))

    #   the anchor VAE encode happens before sampling, not inside the hook
    vae_counted = stubs.FakeVAE(LATENT_DIM["krea2"])
    p = stubs.FakeP(FORMATS["krea2"], vae_counted)
    v = base_values(); v.update(contrast=0.4, color_shift=True, color_shift_amount=0.4, red=0.3)
    script.before_process_batch(p, *ui_args(mod, v))
    script.process_before_every_sampling(p, *ui_args(mod, v))
    encodes_before = vae_counted.calls
    fn = p.sd_model.forge_objects.unet.post_cfg[-1]
    for i in range(len(sigmas) - 1):
        fn({"denoised": stubs.make_latent(seed=5, five_d=True), "sigma": sigmas[i:i + 1],
            "model_options": {"transformer_options": {"sampling_sigmas": sigmas}}})
    check("colour anchors are encoded before sampling, never inside the hook",
          encodes_before == 2 and vae_counted.calls == encodes_before,
          f"{encodes_before} encodes up front, {vae_counted.calls - encodes_before} during")

    #   infotext round-trip
    v = base_values()
    v.update(exposure=0.37, temperature=-0.5, masking=True, mask_axis="hue",
             mask_mode="protect range", more_colors=True, lab_a=0.25, mode="legacy",
             smooth=False, apply_to_hr=False)
    blob = spec.to_infotext(v)
    back = spec.from_infotext(blob)
    diffs = [k for k in v if back.get(k) != v[k]]
    check("infotext round-trips every non-default value", not diffs,
          f"blob={blob!r}" if not diffs else f"differs: {diffs}")

    sparse = spec.to_infotext(base_values())
    check("infotext at stock settings carries only the version tag",
          sparse == spec.VERSION, repr(sparse))

    #   the stale-paste trap: B must fully replace A, not inherit its leftovers
    a = base_values(); a.update(exposure=0.8, vibrance=0.6)
    b = base_values(); b.update(temperature=0.3)
    getter = spec.field_getter("exposure")
    check("pasting a second look resets fields the first one set",
          getter({spec.INFOTEXT_KEY: spec.to_infotext(b)}) == P.BY_NAME["exposure"]["default"]
          and getter({spec.INFOTEXT_KEY: spec.to_infotext(a)}) == 0.8)

    check("unknown/garbage infotext degrades to defaults",
          spec.from_infotext("v9;not_a_param=3;exposure=oops")["exposure"] == 0.0)

    #   and the payoff: script output == node output on identical inputs
    print("\nTier 3b — Forge hook output vs the ComfyUI node's, same inputs")
    current = stubs.load_current_nodes()
    for family, latent_format in FORMATS.items():
        worst, failed = 0.0, None
        for label, v in scenarios():
            vae_a = stubs.FakeVAE(LATENT_DIM[family])
            vae_b = stubs.FakeVAE(LATENT_DIM[family])
            five_d = LATENT_DIM[family] == 3
            x = stubs.make_latent(seed=23, five_d=five_d)
            sigmas = stubs.make_sigmas(8)

            node_out = run_node_post_cfg(current, chain_via_nodes(current, v), x, sigmas,
                                         latent_format, vae_a)

            p = stubs.FakeP(latent_format, vae_b)
            script.before_process_batch(p, *ui_args(mod, v))
            script.process_before_every_sampling(p, *ui_args(mod, v))
            hooks = p.sd_model.forge_objects.unet.post_cfg
            if not hooks:
                failed = f"{label}: script registered no hook"
                break
            fn = hooks[-1]
            for i in range(len(sigmas) - 1):
                got = fn({"denoised": x, "sigma": sigmas[i:i + 1],
                          "model_options": {"transformer_options": {"sampling_sigmas": sigmas}}})
                d = (got.double() - node_out[i].double()).abs().max().item()
                worst = max(worst, d)
                if d > 1e-6:
                    failed = f"{label} step {i}: max|diff|={d:.3e}"
                    break
            if failed:
                break
        check(f"{family} — {len(scenarios())} scenarios match the node",
              failed is None, failed or f"max|diff|={worst:.1e}")


# ---------------------------------------------------------------------------
# Tier 4 — parameter table vs INPUT_TYPES
# ---------------------------------------------------------------------------

def tier4_params_drift():
    print("\nTier 4 — parameter table vs nodes.py INPUT_TYPES")
    from lib_colorcraft import params as P

    current = stubs.load_current_nodes()
    #   Widget declarations only. A `required` entry whose type is one of the
    #   COLORCRAFT_* link types is a graph wire (MaskBlur's `mask`, MaskCombine's
    #   `mask_a`/`mask_b`), which the flat panel replaces with layout rather than
    #   with a control.
    WIDGET_TYPES = {"FLOAT", "INT", "BOOLEAN", "STRING"}
    upstream = {}
    for cls in (current.ColorcraftAdvanced, current.ColorcraftSchedule, current.ColorcraftMasking,
                current.ColorcraftMaskBlur, current.ColorcraftMaskCombine,
                current.ColorcraftMaskPreview):
        for name, decl in cls.INPUT_TYPES()["required"].items():
            head = decl[0]
            if isinstance(head, str) and head not in WIDGET_TYPES:
                continue  # a link input, not a widget
            upstream.setdefault(name, decl)

    alias = {v: k for k, v in P.UPSTREAM_ALIASES.items()}

    bad_default, bad_range, unexpected = [], [], []
    for name, decl in upstream.items():
        table_name = alias.get(name, name)
        if name in P.DROPPED:
            continue
        row = P.BY_NAME.get(table_name)
        if row is None:
            unexpected.append(name)
            continue

        if isinstance(decl, tuple) and isinstance(decl[0], list):
            if row["default"] != decl[0][0] and row["kind"] == P.CHOICE:
                bad_default.append((name, row["default"], decl[0][0]))
            continue

        opts = decl[1] if len(decl) > 1 else {}
        if "default" in opts and row["default"] != opts["default"]:
            bad_default.append((name, row["default"], opts["default"]))

        up = (float(opts.get("min", 0)), float(opts.get("max", 0)))
        here = (float(row["minimum"]), float(row["maximum"])) if row["minimum"] is not None else up
        if up != here:
            expected = P.RANGE_DEVIATIONS.get(table_name)
            if expected != (up, here):
                bad_range.append((name, up, here, expected))

    check("every shared parameter keeps its upstream default", not bad_default, str(bad_default))
    check("every range change is a recorded, deliberate deviation", not bad_range, str(bad_range))
    check("no upstream parameter is silently missing from the table", not unexpected, str(unexpected))

    covered = set(P.RANGE_DEVIATIONS) & set(P.BY_NAME)
    check("RANGE_DEVIATIONS has no stale entries",
          covered == set(P.RANGE_DEVIATIONS),
          str(set(P.RANGE_DEVIATIONS) - covered))

    dup = [n for n in P.ordered_names() if P.ordered_names().count(n) > 1]
    check("parameter names are unique", not dup, str(dup))

    from lib_colorcraft import spec
    missing = [n for n in spec._ADVANCED_KEYS if n not in P.BY_NAME]
    check("every key engine.apply_chain reads exists in the table", not missing, str(missing))


# ---------------------------------------------------------------------------

def main():
    print(f"Colorcraft port harness — latent formats: {FORMAT_SOURCE}")
    print(f"torch {torch.__version__}, {len(scenarios())} scenarios")

    tier1_refactor_equivalence()
    tier2_translation()
    tier3_forge_script()
    tier4_params_drift()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
