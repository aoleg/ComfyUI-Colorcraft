"""The Phase 1 parameter table — one declarative source of truth.

Data, not code, on purpose. Three consumers, only one of which runs Python:

  * the Forge Neo script builds its whole `ui()` from this;
  * `spec.py` reads the defaults to keep infotext sparse and to round-trip it;
  * the future SwarmUI extension is C# and drives the ComfyUI backend by
    emitting graph nodes, so what it can reuse is this table (via a JSON dump),
    not `engine.py`.

`nodes.py` keeps its own hand-written `INPUT_TYPES` for now — no reason to
rewrite working code to prove a refactor. `tests/test_params_drift.py` asserts
the two agree on names, kinds and defaults so they can't silently diverge
before Phase 2 merges them.

TIERING — this deliberately mirrors `ColorcraftAdvanced`'s own gating rather
than inventing a new one. Four of the six gates below (`advanced`,
`more_colors`, `color_shift`, `dev`) are real branches in the upstream math, not
presentation: flipping them off genuinely changes what runs. `masking`/`mask_b`
are ours, standing in for "is a mask node wired up".

RANGES — the ±10 sliders upstream are 2000 positions of Gradio drag for what the
README says should be a gentle adjustment ("be gentle, or give the model time to
heal"). Narrowed to ±3 here; see RANGE_DEVIATIONS for the full list. Every
component is built with `do_not_save_to_config = True`, so unlike
`knowledge_sigmas.md` §5.1's cautionary tale a later recalibration actually
reaches people who already ran an older build.
"""

from . import core


FLOAT, BOOL, CHOICE = "float", "bool", "choice"


def _p(name, kind, default, group, gate=None, label=None, info=None,
       minimum=None, maximum=None, step=0.01, choices=None):
    return {
        "name": name, "kind": kind, "default": default, "group": group, "gate": gate,
        "label": label or name.replace("_", " ").capitalize(), "info": info,
        "minimum": minimum, "maximum": maximum, "step": step, "choices": choices,
    }


#   the everyday amount axes; ±3 rather than upstream's ±10
AMT = dict(minimum=-3.0, maximum=3.0, step=0.01)
#   schedule amounts, in the same units as `strength`
SCH = dict(minimum=-2.0, maximum=2.0, step=0.01)

_MASK_LEAF_INFO = {
    "mode": "highs/lows gate one side of the axis; range keys a band; split is signed (-1..1) and ignores strength",
    "axis": "what the mask reads off the latent — a live measurement, not a painted region",
    "center": "where on the axis the gate sits; for the hue axis this is in units of pi radians, elsewhere normalised to ±1",
    "hardness": "edge falloff; 0 is a wide gradient, high is nearly a hard cut",
    "strength": "blends the finished mask toward fully-open; 0 disables it",
    "width": "band half-width, range modes only",
}


def _mask_leaf(prefix, group, gate):
    """Six controls per leaf. `prefix` is "" for mask A, "b_" for mask B."""
    tag = "" if not prefix else " (B)"
    return [
        _p(f"mask_{prefix}mode", CHOICE, "highs", group, gate,
           label=f"Mode{tag}", info=_MASK_LEAF_INFO["mode"], choices=core.MASK_MODE_OPTIONS),
        _p(f"mask_{prefix}axis", CHOICE, "exposure", group, gate,
           label=f"Axis{tag}", info=_MASK_LEAF_INFO["axis"], choices=core.MASK_AXIS_OPTIONS),
        _p(f"mask_{prefix}center", FLOAT, 0.0, group, gate,
           label=f"Center{tag}", info=_MASK_LEAF_INFO["center"], minimum=-2.0, maximum=2.0),
        _p(f"mask_{prefix}hardness", FLOAT, 1.0, group, gate,
           label=f"Hardness{tag}", info=_MASK_LEAF_INFO["hardness"], minimum=0.0, maximum=20.0, step=0.05),
        _p(f"mask_{prefix}strength", FLOAT, 1.0, group, gate,
           label=f"Strength{tag}", info=_MASK_LEAF_INFO["strength"], minimum=0.0, maximum=1.0),
        _p(f"mask_{prefix}width", FLOAT, 0.0, group, gate,
           label=f"Width{tag}", info=_MASK_LEAF_INFO["width"], minimum=0.0, maximum=2.0),
    ]


PARAMS = [
    # -- schedule ---------------------------------------------------------
    _p("strength", FLOAT, 1.0, "schedule", label="Strength",
       info="peak amount the whole modifier is multiplied by; negative inverts every axis", **SCH),
    _p("start", FLOAT, 0.5, "schedule", label="Start",
       info="fraction of the step progression the ramp begins at", minimum=0.0, maximum=1.0),
    _p("end", FLOAT, 0.75, "schedule", label="End",
       info="and where it returns to zero; early = steers the generation, late = behaves like colour correction",
       minimum=0.0, maximum=1.0),

    _p("advanced", BOOL, False, "gate", label="Schedule shaping"),
    _p("bias", FLOAT, 0.5, "shaping", "advanced", label="Bias",
       info="where the peak sits between start and end", minimum=0.0, maximum=1.0),
    _p("smooth", BOOL, True, "shaping", "advanced", label="Smooth",
       info="cosine ease instead of a linear ramp"),
    _p("exponent", FLOAT, 0.0, "shaping", "advanced", label="Exponent",
       info="curve shape; positive holds low longer, negative rises early", minimum=-10.0, maximum=10.0),
    _p("start_off", FLOAT, 0.0, "shaping", "advanced", label="Start offset",
       info="amount held before the ramp begins, instead of zero", **SCH),
    _p("end_off", FLOAT, 0.0, "shaping", "advanced", label="End offset",
       info="amount held after the ramp ends; keep this at 0 on the last steps unless the edit is very mild", **SCH),

    # -- luma -------------------------------------------------------------
    _p("exposure", FLOAT, 0.0, "luma", label="Exposure",
       info="shifts the whole image along the model's own brightness axis", **AMT),
    _p("tone_compression", FLOAT, 0.0, "luma", label="Tone compression",
       info="1 flattens tone toward neutral, -1 expands it", minimum=-1.0, maximum=1.0),

    # -- punch ------------------------------------------------------------
    _p("contrast", FLOAT, 0.0, "punch", label="Contrast",
       info="whole-latent contrast; the one vector-free axis, works on any model", **AMT),
    _p("clarity", FLOAT, 0.0, "punch", label="Clarity",
       info="mid-frequency local contrast", **AMT),
    _p("sharpness", FLOAT, 0.0, "punch", label="Sharpness",
       info="fine detail and texture; can genuinely add it back, unlike a post filter", **AMT),

    # -- chroma -----------------------------------------------------------
    _p("temperature", FLOAT, 0.0, "chroma", label="Temperature",
       info="warm/cool along the model's own axis, not an RGB channel mix", **AMT),
    _p("tint", FLOAT, 0.0, "chroma", label="Tint", info="green/magenta", **AMT),
    _p("vibrance", FLOAT, 0.0, "chroma", label="Vibrance",
       info="protects already-saturated pixels; pushes near-neutral ones harder", **AMT),
    _p("saturation", FLOAT, 0.0, "chroma", label="Saturation",
       info="plain linear chroma gain, no protection", minimum=-1.0, maximum=3.0),
    _p("chroma_contrast", FLOAT, 0.0, "chroma", label="Chroma contrast",
       info="contrast on chroma magnitude, pivoted at the centre below", **AMT),
    _p("chroma_center", FLOAT, 0.5, "chroma", label="Chroma center",
       info="pivot, as a fraction of the model's max chroma; only active when chroma contrast is non-zero",
       minimum=0.0, maximum=1.0),

    # -- chroma plus ------------------------------------------------------
    _p("more_colors", BOOL, False, "gate", label="More colors"),
    _p("temp_plus_tint", FLOAT, 0.0, "more_colors", "more_colors", label="Temp + tint", **AMT),
    _p("temp_minus_tint", FLOAT, 0.0, "more_colors", "more_colors", label="Temp - tint", **AMT),
    _p("lab_a", FLOAT, 0.0, "more_colors", "more_colors", label="Lab a", **AMT),
    _p("lab_b", FLOAT, 0.0, "more_colors", "more_colors", label="Lab b", **AMT),
    _p("lab_a_plus_b", FLOAT, 0.0, "more_colors", "more_colors", label="Lab a + b", **AMT),
    _p("lab_a_minus_b", FLOAT, 0.0, "more_colors", "more_colors", label="Lab a - b", **AMT),

    # -- colour shift -----------------------------------------------------
    _p("color_shift", BOOL, False, "gate", label="Color shift"),
    _p("color_shift_amount", FLOAT, 0.0, "color_shift", "color_shift", label="Amount",
       info="how far to pull the latent toward the colour below; 0 is a no-op", **AMT),
    _p("mode", CHOICE, "default", "color_shift", "color_shift", label="Mode",
       info="default eases toward the colour and preserves range; legacy is a straight lerp",
       choices=["default", "legacy"]),
    _p("red", FLOAT, 0.0, "color_shift", "color_shift", label="Red", minimum=-2.0, maximum=2.0),
    _p("green", FLOAT, 0.0, "color_shift", "color_shift", label="Green", minimum=-2.0, maximum=2.0),
    _p("blue", FLOAT, 0.0, "color_shift", "color_shift", label="Blue", minimum=-2.0, maximum=2.0),
    _p("brightness", FLOAT, 0.0, "color_shift", "color_shift", label="Brightness", minimum=-2.0, maximum=2.0),

    # -- masking ----------------------------------------------------------
    _p("masking", BOOL, False, "gate", label="Masking"),
    *_mask_leaf("", "mask_a", "masking"),
    _p("mask_combine", BOOL, False, "mask_b_gate", "masking", label="Combine with a second mask"),
    _p("mask_operation", CHOICE, "and", "mask_b", "mask_b", label="Operation",
       info="fuzzy set logic — reduces to ordinary boolean at the 0/1 extremes",
       choices=core.MASK_COMBINE_OPTIONS),
    *_mask_leaf("b_", "mask_b", "mask_b"),
    _p("mask_blur_radius", FLOAT, 0.0, "mask_blur", "masking", label="Blur radius",
       info="in decoded-image pixels; spreads the mask past the pixels that satisfied it. Applied after the combine",
       minimum=0.0, maximum=160.0, step=0.1),
    _p("mask_spread", FLOAT, 0.0, "mask_blur", "masking", label="Spread",
       info="grows (+) or shrinks (-) the blurred mask's coverage rather than just fading it",
       minimum=-3.0, maximum=3.0),
    _p("mask_preview", BOOL, False, "mask_preview", "masking", label="Preview mask instead of applying",
       info="paints the mask over the image instead of running the edits — use it to tune, then switch back"),
    _p("mask_preview_color", CHOICE, "red", "mask_preview", "masking", label="Preview color",
       choices=["red", "green", "blue", "white", "black"]),

    # -- dev --------------------------------------------------------------
    _p("dev", BOOL, False, "gate", label="Advanced"),
    _p("recenter_override", BOOL, False, "dev", "dev", label="Override recenter"),
    _p("recenter", FLOAT, 0.5, "dev", "dev", label="Recenter",
       info="how much chroma drift is corrected after a vibrance/saturation move; 1 also removes deliberate colour casts",
       minimum=0.0, maximum=1.0),
    _p("max_chroma_override", BOOL, False, "dev", "dev", label="Override max chroma"),
    _p("max_chroma", FLOAT, 2.5, "dev", "dev", label="Max chroma",
       info="the model's chroma ceiling; vibrance and the saturation mask are measured against it",
       minimum=0.0, maximum=10.0),
    _p("chroma_plane_override", BOOL, False, "dev", "dev", label="Override chroma plane"),
    _p("chroma_plane", CHOICE, "temp_tint", "dev", "dev", label="Chroma plane",
       info="which axis pair every chroma operation and the hue/saturation masks work in",
       choices=["temp_tint", "lab"]),
]


#   Forge-only controls; not part of the shared chain, so kept out of PARAMS.
RUNTIME_PARAMS = [
    _p("apply_to_hr", BOOL, True, "runtime", label="Apply to the Hires. fix pass"),
    _p("debug", BOOL, False, "runtime", label="Debug logging",
       info="one-shot per run: resolved VAE family, the sigma schedule, and the step each modifier value landed on"),
]


#   Deliberate departures from `nodes.py`'s widget ranges, asserted by the drift
#   test so a future upstream change shows up as a decision rather than a diff.
RANGE_DEVIATIONS = {
    "exposure": ((-10.0, 10.0), (-3.0, 3.0)),
    "contrast": ((-10.0, 10.0), (-3.0, 3.0)),
    "clarity": ((-10.0, 10.0), (-3.0, 3.0)),
    "sharpness": ((-10.0, 10.0), (-3.0, 3.0)),
    "temperature": ((-10.0, 10.0), (-3.0, 3.0)),
    "tint": ((-10.0, 10.0), (-3.0, 3.0)),
    "vibrance": ((-10.0, 10.0), (-3.0, 3.0)),
    "saturation": ((-1.0, 10.0), (-1.0, 3.0)),
    "chroma_contrast": ((-10.0, 10.0), (-3.0, 3.0)),
    "temp_plus_tint": ((-10.0, 10.0), (-3.0, 3.0)),
    "temp_minus_tint": ((-10.0, 10.0), (-3.0, 3.0)),
    "lab_a": ((-10.0, 10.0), (-3.0, 3.0)),
    "lab_b": ((-10.0, 10.0), (-3.0, 3.0)),
    "lab_a_plus_b": ((-10.0, 10.0), (-3.0, 3.0)),
    "lab_a_minus_b": ((-10.0, 10.0), (-3.0, 3.0)),
    "color_shift_amount": ((-10.0, 10.0), (-3.0, 3.0)),
    "start_off": ((-10.0, 10.0), (-2.0, 2.0)),
    "end_off": ((-10.0, 10.0), (-2.0, 2.0)),
    "mask_hardness": ((0.0, 100.0), (0.0, 20.0)),
}

#   In `nodes.py` these live on ColorcraftMasking/MaskBlur without the prefix the
#   flat Forge panel needs. Maps table name -> upstream widget name.
UPSTREAM_ALIASES = {
    "mask_blur_radius": "radius",
    "mask_spread": "spread",
    "mask_operation": "operation",
    "mask_preview_color": "color",
}

#   Present upstream, deliberately absent here: purely a LiteGraph artifact that
#   drew tick marks on the JS schedule plot. It was never read server-side, and
#   Forge already knows the real step count.
DROPPED = {"plot_steps"}


BY_NAME = {p["name"]: p for p in PARAMS}
ALL_BY_NAME = {p["name"]: p for p in PARAMS + RUNTIME_PARAMS}


def defaults():
    return {p["name"]: p["default"] for p in PARAMS}


def in_group(group):
    return [p for p in PARAMS if p["group"] == group]


def ordered_names():
    """The canonical argument order the Forge script's `ui()` returns and every
    hook receives. Runtime params come last so adding a chain param can't shift
    them."""
    return [p["name"] for p in PARAMS] + [p["name"] for p in RUNTIME_PARAMS]
