"""Flat UI values <-> the modifier-chain dicts `engine.py` consumes, plus the
infotext round-trip.

Every Forge component is built with `do_not_save_to_config = True` (too many of
them, and they are per-image settings rather than preferences), so
`ui-config.json` holds nothing and **infotext is the only persistence path**.
That raises the stakes on it: 60-odd separate `parameters` keys per PNG would be
unreadable, so everything goes into one compact key with defaults omitted:

    Colorcraft: "v1;exposure=0.3;start=0.4;mask_axis=hue"

Sparse serialisation has a trap. If each component were bound to its own
infotext key, pasting look B after look A would leave A's non-default values
standing wherever B is silent. `field_getter` closes that by decoding the whole
blob and returning the parameter's *default* when the key is absent, so a paste
always fully replaces the previous state.
"""

from . import params as P


VERSION = "v1"

INFOTEXT_KEY = "Colorcraft"


# ---------------------------------------------------------------------------
# UI values -> modifier chain
# ---------------------------------------------------------------------------

_SCHEDULE_KEYS = ("strength", "start", "end", "bias", "exponent", "start_off", "end_off", "smooth")

_ADVANCED_KEYS = (
    "exposure", "tone_compression",
    "contrast", "clarity", "sharpness",
    "temperature", "tint", "vibrance", "saturation", "chroma_contrast", "chroma_center",
    "more_colors", "temp_plus_tint", "temp_minus_tint",
    "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b",
    "color_shift", "color_shift_amount", "mode", "red", "green", "blue", "brightness",
    "recenter_override", "recenter", "max_chroma_override", "max_chroma",
    "chroma_plane_override", "chroma_plane",
)

#   ColorcraftMaskPreview's fixed colours.
PREVIEW_COLORS = {
    "red": (0.5, -0.5, -0.5),
    "green": (-0.5, 0.5, -0.5),
    "blue": (-0.5, -0.5, 0.5),
    "white": (0.5, 0.5, 0.5),
    "black": (-0.5, -0.5, -0.5),
}


def build_schedule(v):
    """`advanced` off zeroes the three shaping amounts, exactly as
    `ColorcraftAdvanced.make` does — the gate is real, not cosmetic."""
    sched = {k: v[k] for k in _SCHEDULE_KEYS}
    if not v.get("advanced", False):
        sched["exponent"] = 0.0
        sched["start_off"] = 0.0
        sched["end_off"] = 0.0
    return sched


def _mask_leaf(v, prefix):
    return {
        "mask_mode": v[f"mask_{prefix}mode"],
        "mask_axis": v[f"mask_{prefix}axis"],
        "mask_center": v[f"mask_{prefix}center"],
        "mask_hardness": v[f"mask_{prefix}hardness"],
        "mask_width": v[f"mask_{prefix}width"],
        "mask_strength": v[f"mask_{prefix}strength"],
    }


def build_mask(v):
    """Phase 1's fixed shape: up to two leaves, one combine, one blur applied to
    the combined result. Upstream's graph can blur either leaf independently and
    chain combines for 3+ masks; that needs the import/export field planned for
    a later phase."""
    if not v.get("masking", False):
        return None

    mask = _mask_leaf(v, "")
    if v.get("mask_combine", False):
        mask = {"operation": v["mask_operation"], "a": mask, "b": _mask_leaf(v, "b_")}
    if v.get("mask_blur_radius", 0.0) > 0 or v.get("mask_spread", 0.0) != 0:
        mask = {"blur": v["mask_blur_radius"], "spread": v["mask_spread"], "a": mask}
    return mask


def build_chain(v):
    """One `advanced` entry — Phase 1 is a single modifier, so the operation
    order is the fixed upstream one and there is nothing to disambiguate.

    With `mask_preview` on, the entry is replaced by the same fixed-colour
    `shift` that `ColorcraftMaskPreview` emits, pinned to the last step, so the
    mask is painted onto the image instead of gating an edit."""
    mask = build_mask(v)

    if v.get("masking", False) and v.get("mask_preview", False):
        r, g, b = PREVIEW_COLORS[v["mask_preview_color"]]
        return [{
            "kind": "shift",
            "params": {"color_shift_amount": 1.0, "mode": "legacy",
                       "red": r, "green": g, "blue": b, "brightness": 0.0},
            "mask": mask,
            "schedule": {"strength": 1.0, "start": 1.0, "end": 1.0, "bias": 0.5,
                         "exponent": 0.0, "start_off": 0.0, "end_off": 0.0, "smooth": True},
        }]

    return [{
        "kind": "advanced",
        "params": {k: v[k] for k in _ADVANCED_KEYS},
        "mask": mask,
        "schedule": build_schedule(v),
    }]


def is_no_op(v):
    """True when the chain would leave every step untouched, so the script can
    skip patching the UNet entirely rather than registering a hook that does
    nothing."""
    if v.get("masking", False) and v.get("mask_preview", False):
        return False
    if v.get("strength", 0.0) == 0.0:
        return True

    amounts = ["exposure", "tone_compression", "contrast", "clarity", "sharpness",
               "temperature", "tint", "vibrance", "saturation", "chroma_contrast"]
    if v.get("more_colors"):
        amounts += ["temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b",
                    "lab_a_plus_b", "lab_a_minus_b"]
    if v.get("color_shift"):
        amounts += ["color_shift_amount"]
    return all(v.get(name, 0.0) == 0.0 for name in amounts)


# ---------------------------------------------------------------------------
# Infotext
# ---------------------------------------------------------------------------

def _fmt(value, kind):
    if kind == P.BOOL:
        return "1" if value else "0"
    if kind == P.FLOAT:
        return f"{round(float(value), 6):g}"
    return str(value)


def _parse(text, kind, fallback):
    try:
        if kind == P.BOOL:
            return text.strip() in ("1", "true", "True", "yes")
        if kind == P.FLOAT:
            return float(text)
        return text
    except (TypeError, ValueError):
        return fallback


def to_infotext(v):
    """Non-default values only, in table order. Returns None when nothing
    deviates, so a run at stock settings writes just `Colorcraft: "v1"` rather
    than a wall of zeroes."""
    parts = [VERSION]
    for p in P.PARAMS + P.RUNTIME_PARAMS:
        name = p["name"]
        if name not in v:
            continue
        if v[name] == p["default"]:
            continue
        parts.append(f"{name}={_fmt(v[name], p['kind'])}")
    return ";".join(parts)


def from_infotext(blob):
    """Always returns a complete value dict: table defaults, overlaid with
    whatever the blob carried. Unknown keys (a newer build's, or junk) are
    ignored rather than raising."""
    values = P.defaults()
    values.update({p["name"]: p["default"] for p in P.RUNTIME_PARAMS})
    if not blob:
        return values

    for chunk in str(blob).split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue  # the leading version tag, or a stray separator
        name, _, raw = chunk.partition("=")
        p = P.ALL_BY_NAME.get(name.strip())
        if p is None:
            continue
        parsed = _parse(raw, p["kind"], p["default"])
        if p["kind"] == P.CHOICE and parsed not in p["choices"]:
            parsed = p["default"]
        values[name.strip()] = parsed
    return values


def field_getter(name):
    """An `infotext_fields` callable for one component. Decodes the whole blob
    and falls back to the table default, so a paste replaces every field rather
    than leaving stale non-defaults behind (see the module docstring)."""
    default = P.ALL_BY_NAME[name]["default"]

    def get(d):
        blob = d.get(INFOTEXT_KEY)
        if blob is None:
            return default
        return from_infotext(blob).get(name, default)

    return get
