"""Fake `comfy`, `modules` and `gradio` packages, plus the fixtures both sides
of the comparison run against.

Nothing here needs a GPU, a model, ComfyUI or Forge. The point (per
knowledge_skimmed_cfg.md §6) is that both reference implementations are plain
Python files on disk, so "the math looks right" can be replaced with "the math
is the same math" in about two seconds.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# comfy
# ---------------------------------------------------------------------------

def install_comfy():
    if "comfy" in sys.modules:
        return sys.modules["comfy"]

    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    model_patcher = types.ModuleType("comfy.model_patcher")
    model_management = types.ModuleType("comfy.model_management")

    class KSAMPLER:
        def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
            self.sampler_function = sampler_function
            self.extra_options = extra_options or {}
            self.inpaint_options = inpaint_options or {}

    def set_model_options_post_cfg_function(model_options, fn, disable_cfg1_optimization=False):
        model_options = dict(model_options)
        model_options["sampler_post_cfg_function"] = \
            list(model_options.get("sampler_post_cfg_function", [])) + [fn]
        return model_options

    samplers.KSAMPLER = KSAMPLER
    model_patcher.set_model_options_post_cfg_function = set_model_options_post_cfg_function
    model_management.get_torch_device = lambda: torch.device("cpu")

    comfy.samplers = samplers
    comfy.model_patcher = model_patcher
    comfy.model_management = model_management
    for name, mod in [("comfy", comfy), ("comfy.samplers", samplers),
                      ("comfy.model_patcher", model_patcher),
                      ("comfy.model_management", model_management)]:
        sys.modules[name] = mod
    return comfy


# ---------------------------------------------------------------------------
# Real latent formats, loaded straight out of the Forge checkout
# ---------------------------------------------------------------------------

def load_latent_formats(forge_root):
    """The genuine `Wan21`/`Flux` classes rather than look-alikes, so the family
    lookup is tested against the class names it will really see, and
    `process_in` against its real broadcasting shapes."""
    path = Path(forge_root) / "modules_forge" / "packages" / "huggingface_guess" / "latent.py"
    if not path.is_file():
        return None
    spec_ = importlib.util.spec_from_file_location("_cc_latent", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


class ToyLatentFormat:
    """Fallback with the same class *name* as the real thing, used only if the
    Forge checkout isn't next door."""
    scale_factor = 0.7
    shift_factor = 0.1

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor


class ToyIdentityLatentFormat(ToyLatentFormat):
    """Flux2's `process_in`/`process_out` are the identity (`latent.py:161`),
    which is the whole reason it needs no anchor conversion."""

    def process_in(self, latent):
        return latent


def make_toy(name, identity=False):
    base = ToyIdentityLatentFormat if identity else ToyLatentFormat
    return type(name, (base,), {})()


def synthetic_basis(channels=128, seed=1234, subpixels=1, norms=None):
    """A stand-in vector file for a family whose real one has not been derived
    yet, so the 23 scenarios can still run at its channel count and latent
    format.

    `subpixels > 1` replicates each direction across a channel's sub-pixel slots,
    matching how a packed family's real vectors are built — otherwise the
    fixture would be the one thing the shipped file is guaranteed not to be."""
    from lib_colorcraft import core

    g = torch.Generator().manual_seed(seed)
    norms = norms or {}
    out = {}
    for i, axis in enumerate(core.PRIMITIVE_AXES):
        v = torch.randn(channels // subpixels, generator=g)
        if subpixels > 1:
            v = v.view(-1, 1).expand(-1, subpixels).reshape(-1)
        out[axis] = (v / v.norm() * norms.get(axis, 1.0 + 0.1 * i)).contiguous()
    for name, (a, b, sign) in core.DIAGONAL_AXES.items():
        va, vb = out[a], out[b]
        d = va / va.norm() + sign * (vb / vb.norm())
        out[name] = (d / d.norm() * (0.5 * (va.norm() + vb.norm()))).contiguous()
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeVAE:
    """`encode` takes [B,H,W,3] in 0..1 on both backends
    (`backend/patcher/vae.py:253`), so one fake serves both. Deterministic and
    colour-dependent, so a mis-wired anchor shows up as a numeric difference
    rather than a coincidental match."""

    def __init__(self, latent_dim=2, channels=16):
        self.latent_dim = latent_dim
        self.latent_channels = channels
        self.first_stage_model = object()  # identity key for the Forge anchor cache
        self.calls = 0

    def encode(self, img):
        self.calls += 1
        C = self.latent_channels
        mean = img.reshape(-1, img.shape[-1]).mean(0)              # [3]
        mix = torch.linspace(-1.0, 1.0, C * 3).reshape(C, 3)
        per_channel = (mix @ mean) + 0.05                          # [C]
        trailing = (1, 8, 8) if self.latent_dim == 3 else (8, 8)
        return per_channel.reshape(1, C, *([1] * len(trailing))).expand(1, C, *trailing).contiguous()


class FakeModel:
    """What ComfyUI hands a post-CFG hook as `args["model"]`. Forge's equivalent
    has no `latent_format` at all — that lives on `p.sd_model.model_config` —
    which is exactly the difference the port has to bridge."""

    def __init__(self, latent_format):
        self.latent_format = latent_format


def make_latent(seed, channels=16, h=16, w=16, batch=1, five_d=False, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    shape = (batch, channels, 1, h, w) if five_d else (batch, channels, h, w)
    return torch.randn(shape, generator=g, dtype=dtype)


def make_sigmas(steps=8, flow=True):
    """A flow-matching schedule: starts at exactly 1.0 and ends at 0, which is
    what every model family Colorcraft has vectors for actually produces."""
    if flow:
        return torch.linspace(1.0, 0.0, steps + 1)
    return torch.cat([torch.linspace(14.6, 0.03, steps), torch.zeros(1)])


# ---------------------------------------------------------------------------
# Loading the two node modules
# ---------------------------------------------------------------------------

def load_current_nodes():
    """`nodes.py` uses relative imports now, so it needs a package to live in.
    A synthetic parent whose `__path__` is the repo root is enough
    (knowledge.md §10.5)."""
    install_comfy()
    pkg_name = "_cc_current"
    if pkg_name + ".nodes" in sys.modules:
        return sys.modules[pkg_name + ".nodes"]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(REPO)]
    sys.modules[pkg_name] = pkg

    #   Bind the *already imported* shared core into the synthetic package
    #   instead of letting `from .lib_colorcraft import core` load a second copy
    #   of it. Two copies means two `VECTORS_DIR`s, and a test that re-points one
    #   of them silently compares a configured module against an unconfigured
    #   one — which is how a family with no vector file can "match" by both
    #   sides doing nothing.
    for sub in ("", ".core", ".engine", ".params", ".spec"):
        sys.modules[pkg_name + ".lib_colorcraft" + sub] = \
            importlib.import_module("lib_colorcraft" + sub)

    spec_ = importlib.util.spec_from_file_location(pkg_name + ".nodes", REPO / "nodes.py")
    mod = importlib.util.module_from_spec(spec_)
    sys.modules[pkg_name + ".nodes"] = mod
    spec_.loader.exec_module(mod)
    return mod


def load_reference_nodes(path):
    """The pre-refactor `nodes.py`, recovered from git. Self-contained — no
    relative imports — so it loads directly; only `VECTORS_DIR` needs
    re-pointing, since it was derived from `__file__`."""
    install_comfy()
    spec_ = importlib.util.spec_from_file_location("_cc_reference_nodes", path)
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["_cc_reference_nodes"] = mod
    spec_.loader.exec_module(mod)
    mod.VECTORS_DIR = str(REPO / "vectors")
    return mod


# ---------------------------------------------------------------------------
# gradio / modules, for the Forge script
# ---------------------------------------------------------------------------

class _Component:
    def __init__(self, value=None, **kwargs):
        self.__dict__.update(kwargs)
        self.value = kwargs.get("value", value)
        self.visible = kwargs.get("visible", True)

    def change(self, *a, **kw):
        return None

    def click(self, *a, **kw):
        return None


class _Ctx:
    def __init__(self, *a, **kw):
        self.__dict__.update(kw)
        self.visible = kw.get("visible", True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def install_gradio():
    if "gradio" in sys.modules:
        return sys.modules["gradio"]
    gr = types.ModuleType("gradio")
    gr.Slider = _Component
    gr.Checkbox = _Component
    gr.Dropdown = _Component
    gr.Radio = _Component
    gr.HTML = _Component
    gr.Textbox = _Component
    gr.Button = _Component
    gr.Row = _Ctx
    gr.Group = _Ctx
    gr.Column = _Ctx
    gr.Accordion = _Ctx
    gr.update = lambda **kw: dict(kw)
    sys.modules["gradio"] = gr
    return gr


class _Logger:
    def __init__(self):
        self.lines = []

    def _log(self, level, msg):
        self.lines.append((level, str(msg)))

    def info(self, msg):
        self._log("info", msg)

    def warning(self, msg):
        self._log("warning", msg)

    def error(self, msg):
        self._log("error", msg)

    def texts(self):
        return [m for _, m in self.lines]


LOGGER = _Logger()


def install_modules():
    if "modules" in sys.modules:
        return sys.modules["modules"]
    install_gradio()

    modules = types.ModuleType("modules")
    scripts_mod = types.ModuleType("modules.scripts")
    processing_mod = types.ModuleType("modules.processing")
    ui_components = types.ModuleType("modules.ui_components")
    devices_mod = types.ModuleType("modules.devices")

    class Script:
        AlwaysVisible = object()
        is_txt2img = True
        is_img2img = False

        def __init__(self):
            self.infotext_fields = []

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = Script.AlwaysVisible
    scripts_mod.basedir = lambda: str(REPO)

    processing_mod.logger = LOGGER

    class InputAccordionImpl(_Component):
        def __init__(self, value=None, **kwargs):
            super().__init__(value=value, **kwargs)
            self.accordion = _Ctx(**kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    ui_components.InputAccordion = lambda value=None, **kw: InputAccordionImpl(value=value, **kw)
    ui_components.InputAccordionImpl = InputAccordionImpl

    devices_mod.device = torch.device("cpu")
    devices_mod.cpu = torch.device("cpu")

    shared_mod = types.ModuleType("modules.shared")
    shared_mod.sd_model = None
    shared_mod.opts = types.SimpleNamespace()
    modules.shared = shared_mod
    sys.modules["modules.shared"] = shared_mod

    modules.scripts = scripts_mod
    modules.processing = processing_mod
    modules.ui_components = ui_components
    modules.devices = devices_mod
    for name, mod in [("modules", modules), ("modules.scripts", scripts_mod),
                      ("modules.processing", processing_mod),
                      ("modules.ui_components", ui_components),
                      ("modules.devices", devices_mod)]:
        sys.modules[name] = mod
    return modules


class FakeUnet:
    def __init__(self):
        self.post_cfg = []
        self.model_options = {"transformer_options": {}}

    def clone(self):
        n = FakeUnet()
        n.post_cfg = list(self.post_cfg)
        n.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        return n

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.post_cfg.append(fn)


class FakeForgeObjects:
    def __init__(self, vae, unet):
        self.vae = vae
        self.unet = unet


class FakeSdModel:
    def __init__(self, latent_format, vae):
        self.model_config = types.SimpleNamespace(latent_format=latent_format)
        self.forge_objects = FakeForgeObjects(vae, FakeUnet())


class FakeP:
    """Only the attributes the script actually touches."""

    def __init__(self, latent_format, vae, steps=8, is_hr_pass=False):
        self.sd_model = FakeSdModel(latent_format, vae)
        self.steps = steps
        self.is_hr_pass = is_hr_pass
        self.extra_generation_params = {}
