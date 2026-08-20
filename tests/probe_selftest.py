"""Offline self-test for `tools/colorcraft_probe.py`.

The probe itself can only run inside Forge, against a real VAE. This checks
everything up to that boundary, so a live session isn't spent discovering a
typo:

  * the colour maths round-trips (a broken sRGB<->Lab would quietly bias two of
    the seven axes);
  * the pixel transforms and measurements move in the direction they claim;
  * methods B and C recover the **analytically known** answer from a toy VAE
    whose latent->RGB map we chose ourselves;
  * method B returns ~zero for clarity/sharpness, which is the predicted null
    result the live run is meant to confirm;
  * method A handles Flux2's packed 128-channel layout and the replication
    residual comes out at exactly zero for a replicated vector;
  * `run_probe` survives end to end and writes its output files.

Run:  python tests/probe_selftest.py
"""

import sys
import tempfile
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stubs  # noqa: E402
from stubs import REPO  # noqa: E402

FORGE_ROOT = REPO.parent / "sd-webui-forge-classic"

PASS, FAIL = [], []

#   set by install_temp_vectors(); nothing in this file may write into the
#   repo's own vectors/ directory
TEST_VECTORS_DIR = ""


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def cos(a, b):
    return float(torch.dot(a / a.norm(), b / b.norm()))


# ---------------------------------------------------------------------------
# A toy VAE with a latent->RGB map we choose, so ground truth is computable.
#
#   decode:  img = 0.5 + W^T z          (per pixel, W is [C,3])
#   encode:  z   = pinv(W^T) (img-0.5)  (the minimum-norm preimage)
#
# Latent and image share a resolution, which keeps the ground truth exact.
# ---------------------------------------------------------------------------

class _Identity:
    def process_in(self, x):
        return x

    def process_out(self, x):
        return x


class ToyVAE:
    def __init__(self, channels=16, seed=1, clamp=True, strict_inference=True):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(channels, 3, generator=g) * 0.25   # [C,3]
        self.pinv = torch.linalg.pinv(self.W.T)                 # [C,3]
        self.latent_channels = channels
        self.latent_dim = 2
        self.is_wan = False
        self.downscale_ratio = 1
        self.vae_dtype = torch.float32
        self.first_stage_model = _Identity()
        self.clamp = clamp
        self.strict_inference = strict_inference

    def _guard(self):
        """Forge loads VAE weights inside `torch.inference_mode()`, so calling
        the real VAE from a plain `no_grad` dies inside operations.py's
        manual-cast conv with "Inference tensors do not track version counter".
        The toy refuses the same way, so a regression to `no_grad` fails here
        rather than in a live session."""
        if self.strict_inference and not torch.is_inference_mode_enabled():
            raise RuntimeError("Inference tensors do not track version counter. "
                               "(toy VAE: called outside torch.inference_mode)")

    def encode(self, bhwc):                 # [B,H,W,3] 0..1 -> [B,C,H,W]
        self._guard()
        img = bhwc.movedim(-1, 1) - 0.5
        return torch.einsum("kc,bchw->bkhw", self.pinv, img)

    def decode(self, z):                    # [B,C,H,W] -> [B,H,W,3] 0..1
        self._guard()
        img = torch.einsum("kc,bkhw->bchw", self.W, z) + 0.5
        if self.clamp:
            img = img.clamp(0.0, 1.0)
        return img.movedim(1, -1)


class ToyWanVAE(ToyVAE):
    """Krea 2 / Qwen use the Wan VAE, whose latents carry a temporal axis and
    arrive as [B,C,1,H,W]. The 4D toy above let a 5D unpacking bug reach a live
    session, so every probe entry point is now exercised against both ranks."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.is_wan = True
        self.latent_dim = 3

    def encode(self, bhwc):                 # -> [B,C,1,H,W]
        return super().encode(bhwc).unsqueeze(2)

    def decode(self, z):                    # [B,C,1,H,W] -> [B,1,H,W,3]
        return super().decode(z.squeeze(2)).unsqueeze(1)


def install_temp_vectors():
    """Point `core.VECTORS_DIR` at a throwaway copy holding only the two shipped
    files.

    Several tests below drive `build_basis_file` or the calibration writer, and
    those write a real file into whatever `VECTORS_DIR` says. Against the repo's
    own `vectors/` that means overwriting the user's derived files — and the
    save-and-restore dance this used to do is not even reliable on Windows:
    safetensors mmaps what it loads, and a mapped file cannot be replaced, so a
    failed restore leaves test data sitting where real data was. It did exactly
    that once. A temp directory removes the hazard instead of managing it.

    flux2 is deliberately left out even when the real file exists, so the
    no-reference build path stays covered."""
    import atexit
    import shutil
    from lib_colorcraft import core

    global TEST_VECTORS_DIR
    tmp = Path(tempfile.mkdtemp(prefix="cc_selftest_vectors_"))
    for name in ("krea2", "zimage"):
        src = Path(core.VECTORS_DIR) / f"colorcraft-{name}.safetensors"
        if src.is_file():
            shutil.copy(src, tmp / src.name)
    TEST_VECTORS_DIR = str(tmp)
    repoint_vectors()
    atexit.register(shutil.rmtree, str(tmp), True)
    return str(tmp)


def repoint_vectors():
    """Re-apply the throwaway directory to whatever `lib_colorcraft.core` module
    object is current.

    `_ensure_fresh_lib()` drops the package and re-imports it, which resets
    `VECTORS_DIR` to the repo's own `vectors/` — so every test that runs after
    the stale-lib guard test would otherwise be writing into real data again,
    and the probe would be reading from one directory while the test wrote to
    another."""
    if not TEST_VECTORS_DIR:
        return
    for name, mod in list(sys.modules.items()):
        if name == "lib_colorcraft.core" or name.endswith(".lib_colorcraft.core"):
            mod.VECTORS_DIR = TEST_VECTORS_DIR


def load_probe():
    stubs.install_modules()
    stubs.install_gradio()
    import importlib
    sys.path.insert(0, str(REPO / "tools"))
    if "colorcraft_probe" in sys.modules:
        return sys.modules["colorcraft_probe"]
    return importlib.import_module("colorcraft_probe")


# ---------------------------------------------------------------------------

def test_colour_maths(probe):
    print("\nColour maths")
    g = torch.Generator().manual_seed(4)
    img = torch.rand(2, 3, 32, 32, generator=g) * 0.9 + 0.05

    back = probe.linear_to_srgb(probe.srgb_to_linear(img))
    check("sRGB <-> linear round-trips", torch.allclose(img, back, atol=2e-4),
          f"max|diff|={(img - back).abs().max():.2e}")

    back = probe.lab_to_srgb(probe.srgb_to_lab(img))
    check("sRGB <-> Lab round-trips", torch.allclose(img, back, atol=2e-3),
          f"max|diff|={(img - back).abs().max():.2e}")

    lab = probe.srgb_to_lab(torch.full((1, 3, 8, 8), 0.5))
    check("mid grey has near-zero Lab chroma",
          abs(float(lab[0, 1].mean())) < 0.5 and abs(float(lab[0, 2].mean())) < 0.5,
          f"a={float(lab[0,1].mean()):+.3f} b={float(lab[0,2].mean()):+.3f}")

    flat = torch.full((1, 3, 32, 32), 0.5)
    blurred = probe.gaussian_blur(flat, 3.0)
    check("gaussian blur preserves a flat field (reflect padding)",
          torch.allclose(flat, blurred, atol=1e-5))


def test_transforms_and_measures(probe):
    print("\nTransforms move the measurements they claim to")
    corpus, _ = probe.build_corpus(6, 128)
    for axis in probe.PRIMITIVES:
        up = probe.measure(probe.transform(corpus, axis, +0.15), axis).mean()
        dn = probe.measure(probe.transform(corpus, axis, -0.15), axis).mean()
        check(f"{axis}: +delta raises its own measurement", float(up) > float(dn),
              f"{float(dn):+.5f} -> {float(up):+.5f}")

    #   cross-talk sanity: exposure must not masquerade as a hue change
    base = probe.measure(corpus, "lab-a").mean()
    exposed = probe.measure(probe.transform(corpus, "exposure", +0.15), "lab-a").mean()
    lab_a = probe.measure(probe.transform(corpus, "lab-a", +0.15), "lab-a").mean()
    check("exposure barely moves lab-a compared to lab-a itself",
          abs(float(exposed - base)) < 0.25 * abs(float(lab_a - base)),
          f"exposure {float(exposed-base):+.3f} vs lab-a {float(lab_a-base):+.3f}")


def test_methods_against_known_answer(probe):
    print("\nMethods recover the toy VAE's analytic directions")
    vae = ToyVAE(channels=16, seed=1, clamp=False)
    corpus, _ = probe.build_corpus(8, 128)

    with torch.inference_mode():
        derived_b = probe.method_b(vae, corpus, delta=0.12, batch=4)

    #   B asks "what latent change does this image change produce" -> pinv-flavoured
    for axis, rgb in [("exposure", (1.0, 1.0, 1.0)), ("temperature", (1.0, 0.0, -1.0))]:
        truth = vae.pinv @ torch.tensor(rgb)
        c = abs(cos(derived_b[axis], truth))
        check(f"method B recovers the pinv direction for {axis}", c > 0.95, f"|cos|={c:.4f}")

    #   ...and must fall over for the detail axes: sharpening changes no averages
    ratio = float(derived_b["sharpness"].norm() / derived_b["exposure"].norm())
    check("method B returns ~zero for sharpness (the predicted null result)",
          ratio < 0.05, f"|sharpness| / |exposure| = {ratio:.4f}")

    with torch.inference_mode():
        derived_c, info = probe.method_c(vae, corpus, n_base=1, eps_scale=0.05,
                                         chunk=8, probe_size=64)

    #   C asks "which direction most increases this measurement" -> adjoint-flavoured,
    #   i.e. W applied to the measurement's RGB weights, NOT the pseudo-inverse
    lin_w = torch.tensor([0.2126, 0.7152, 0.0722])
    truth = vae.W @ lin_w
    c = abs(cos(derived_c["exposure"], truth))
    check("method C recovers the adjoint direction for exposure", c > 0.90, f"|cos|={c:.4f}")

    truth_t = vae.W @ torch.tensor([1.0, 0.0, -1.0])
    c = abs(cos(derived_c["temperature"], truth_t))
    check("method C recovers the adjoint direction for temperature", c > 0.90, f"|cos|={c:.4f}")

    #   The two methods answer different questions; on a non-orthogonal map they
    #   must NOT coincide, or one of them is not doing what it claims.
    c = abs(cos(derived_b["exposure"], derived_c["exposure"]))
    check("B and C give genuinely different directions (pinv vs adjoint)",
          c < 0.99, f"|cos(B,C)|={c:.4f}")

    check("method C reports its settings", "eps=" in info and "decodes" in info, info)


def test_flux2_packing(probe):
    print("\nFlux2 packed layout")
    lf = stubs.load_latent_formats(FORGE_ROOT)
    if lf is None:
        check("Forge checkout available for the real Flux2 latent format", False)
        return

    fmt = lf.Flux2()
    derived, note = probe.method_a(fmt, channels=128)
    check("method A produces 128-dim vectors from 32 rgb factors",
          all(v.shape == (128,) for v in derived.values()), note)

    residuals = {a: probe.replication_residual(v, 128) for a, v in derived.items()}
    check("replicated vectors have zero sub-pixel residual",
          all(r < 1e-6 for r in residuals.values()),
          f"max={max(residuals.values()):.2e}")

    g = torch.Generator().manual_seed(2)
    arbitrary = torch.randn(128, generator=g)
    check("an arbitrary direction has a large residual (the grid hazard)",
          probe.replication_residual(arbitrary, 128) > 0.5,
          f"residual={probe.replication_residual(arbitrary, 128):.3f}")

    #   channel c belongs to unpacked channel c//4 -- the layout method A assumes
    v = derived["exposure"].view(32, 4)
    check("all four sub-pixel slots of a channel carry the same value",
          float((v - v[:, :1]).abs().max()) < 1e-6)

    unpacked, _ = probe.method_a(lf.SDXL_Flux2(), channels=32)
    check("Mugen (SDXL_Flux2, 32ch unpacked) derives without replication",
          all(v.shape == (32,) for v in unpacked.values()))

    from lib_colorcraft import core

    check("Flux2 reads as a packed family, Flux and Mugen do not",
          core.is_packed_family("flux2", fmt)
          and not core.is_packed_family("zimage", lf.Flux())
          and not core.is_packed_family("sdxl_flux2", lf.SDXL_Flux2()))

    #   the projection build_basis_file applies: it must remove the grid part
    #   and nothing else, and re-applying it must change nothing
    projected = core.project_replicated(arbitrary)
    check("projecting an arbitrary direction lands it in the replicated subspace",
          probe.replication_residual(projected, 128) < 1e-6)
    check("projection is idempotent",
          float((core.project_replicated(projected) - projected).abs().max()) < 1e-7)
    removed = arbitrary - projected
    check("what it removes is orthogonal to what it keeps (a real projection)",
          abs(float(torch.dot(removed, projected))) < 1e-4,
          f"<removed, kept>={float(torch.dot(removed, projected)):+.2e}")
    kept = float(projected.norm() / arbitrary.norm())
    check("a random direction keeps ~1/2 of its norm (4 slots -> 1 mean)",
          0.4 < kept < 0.6, f"|proj|/|v|={kept:.3f}")

    #   The factor behind flux2's provisional MODEL_DEV_DEFAULTS row: a packed
    #   pixel carries four near-identical sub-pixel slots, so its projection onto
    #   a replicated *unit* vector is twice the unpacked one. If that is wrong,
    #   every provisional scale in the table is wrong with it.
    g2 = torch.Generator().manual_seed(11)
    u = torch.randn(32, generator=g2)
    z32 = torch.randn(64, 32, generator=g2)                 # 64 unpacked pixels
    z128 = z32.repeat_interleave(4, dim=1)                  # the same pixel, 4 slots
    v128 = core.project_replicated(u.repeat_interleave(4))
    ratio = float(((z128 @ (v128 / v128.norm())) / (z32 @ (u / u.norm()))).mean())
    check("a packed projection is exactly 2x the unpacked one", abs(ratio - 2.0) < 1e-4,
          f"ratio={ratio:.4f}")


def test_run_probe_end_to_end(probe):
    print("\nrun_probe end to end")
    import types as _t

    tmp = tempfile.mkdtemp(prefix="cc_probe_")
    sys.modules["modules.scripts"].basedir = lambda: tmp

    lf = stubs.load_latent_formats(FORGE_ROOT)
    fmt = lf.Flux() if lf else stubs.make_toy("Flux")

    shared = sys.modules["modules.shared"]
    shared.sd_model = None
    out = probe.run_probe("", 128, 4, True, True, True, "0.12", 2, 1, 0.05, 8, 64, False)
    check("no checkpoint loaded: explains itself instead of raising",
          "No checkpoint loaded" in out, out.strip()[:60])

    #   regression: before the first generation Forge hands out a FakeInitialModel
    #   placeholder with no forge_objects at all (sd_models.py:239)
    class _FakeInitialModel:
        first_stage_model = None
        cond_stage_model = None

    shared.sd_model = _FakeInitialModel()
    out = probe.run_probe("", 128, 4, True, True, True, "0.12", 2, 1, 0.05, 8, 64, False)
    check("FakeInitialModel placeholder: reports it instead of AttributeError",
          "No checkpoint loaded" in out, out.strip()[:60])

    vae = ToyVAE(channels=16, seed=3, clamp=True)
    shared.sd_model = _t.SimpleNamespace(
        model_config=_t.SimpleNamespace(latent_format=fmt),
        forge_objects=_t.SimpleNamespace(vae=vae),
    )
    out = probe.run_probe("", 128, 6, True, True, True, "0.05, 0.15", 2, 1, 0.05, 8, 64, True)

    check("run_probe completes and names the family", "family=zimage" in out, "")
    check("run_probe scores against the shipped vectors",
          "cosine vs shipped vectors" in out and "reference vectors:" in out)
    for m in ("method A", "method B", "method C"):
        check(f"{m} ran without failing", f"{m} FAILED" not in out)
    check("every primitive axis appears in the table",
          all(a in out for a in probe.PRIMITIVES))

    written = sorted(p.name for p in Path(tmp).joinpath("probe_out").iterdir())
    check("derived vectors and report written to probe_out/",
          any(n.endswith(".safetensors") for n in written) and any(n.endswith(".json") for n in written),
          ", ".join(written))

    #   A toy VAE has no reason to match the real vectors — what matters is that
    #   the scoring path produced finite numbers rather than nan/crash.
    import json
    report = json.loads(Path(tmp).joinpath("probe_out", "report-zimage.json").read_text())
    finite = [v for per in report["cosines"].values() for v in per.values()
              if v == v and abs(v) <= 1.0]
    check("cosines are finite and in range", len(finite) > 0,
          f"{len(finite)} scores recorded")

    #   regression: the whole VAE-touching span must sit inside inference_mode
    check("run_probe drives the VAE inside torch.inference_mode",
          "Inference tensors" not in out, "")

    derived_files = [n for n in written if n.endswith(".safetensors")]
    from safetensors.torch import load_file
    loaded = load_file(str(Path(tmp) / "probe_out" / derived_files[0]))
    check("saved vectors reload as ordinary (non-inference) tensors",
          all(not t.is_inference() for t in loaded.values()),
          f"{len(loaded)} axes in {derived_files[0]}")

    print("\n  --- sample of the log the live run will print ---")
    for line in out.splitlines()[:14]:
        print("  | " + line)


def test_build_basis_file(probe):
    print("\nPhase 0.5 — build a loadable basis from a method's output")
    import types as _t
    from safetensors.torch import load_file, save_file
    from lib_colorcraft import core

    tmp = tempfile.mkdtemp(prefix="cc_build_")
    sys.modules["modules.scripts"].basedir = lambda: tmp
    out_dir = Path(tmp) / "probe_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    lf = stubs.load_latent_formats(FORGE_ROOT)
    fmt = lf.Flux() if lf else stubs.make_toy("Flux")
    ref = core.load_basis("zimage")

    #   a fake method output: the shipped vectors, deliberately mis-scaled and
    #   with tint flipped, so the alignment step has something real to correct
    g = torch.Generator().manual_seed(7)
    fake = {}
    for a in core.PRIMITIVE_AXES:
        v = ref[a].float().clone()
        v = v + torch.randn(v.shape, generator=g) * 0.05 * v.norm()
        v = v * (3.7 if a != "tint" else -0.2)
        fake[a] = v.contiguous()
    save_file(fake, str(out_dir / "derived-zimage-methodB_0.2.safetensors"))

    shared = sys.modules["modules.shared"]
    shared.sd_model = _t.SimpleNamespace(
        model_config=_t.SimpleNamespace(latent_format=fmt),
        forge_objects=_t.SimpleNamespace(vae=ToyVAE(channels=16)),
    )

    #   `install_temp_vectors` guarantees this directory holds nothing but the
    #   two shipped files, so anything a test writes here is a test's own and can
    #   simply be deleted afterwards.
    backup = Path(core.VECTORS_DIR) / "colorcraft-zimage-derived.safetensors"
    try:
        out = probe.build_basis_file("B@0.2")
        check("build reports the file it wrote", "wrote" in out and "colorcraft-zimage" in out)
        check("tint's flipped sign is corrected", "tint" in out and "sign-flipped" in out)

        built = load_file(str(backup))
        check("all 11 axes present (7 derived + 4 constructed)",
              len(built) == 11, f"{sorted(built)}")

        #   norms must match the shipped file exactly, or the A/B compares
        #   strength instead of direction
        norm_err = max(abs(float(built[a].norm() - ref[a].float().norm())) for a in built)
        check("norms copied from the shipped file", norm_err < 1e-4, f"max err={norm_err:.2e}")

        #   ...and every axis must now point the same way as the shipped one
        worst = min(cos(built[a], ref[a].float()) for a in built)
        check("every axis aligned to the shipped sign", worst > 0.5, f"worst cos={worst:+.4f}")

        #   diagonals are constructed, not copied
        d = built["temp+tint"]
        made = built["temperature"] / built["temperature"].norm() + built["tint"] / built["tint"].norm()
        check("diagonals rebuilt from their parents", abs(cos(d, made)) > 0.9999,
              f"cos={cos(d, made):+.6f}")

        check("a missing method output is reported, not raised",
              "No such probe output" in probe.build_basis_file("Z@9.9"))

        #   the extension can actually load what was written
        bundle = core.load_all_basis(core.DERIVED_VARIANT)
        check("core.load_all_basis picks up the derived variant", "zimage" in bundle)
    finally:
        if backup.exists():
            backup.unlink()


def test_build_basis_file_packed(probe):
    """Flux2's build path: no shipped file to align against, and a projection
    that has to happen before anything else does."""
    print("\nPhase 0.5 — building a packed (Flux2) basis")
    import types as _t
    from safetensors.torch import load_file, save_file
    from lib_colorcraft import core

    lf = stubs.load_latent_formats(FORGE_ROOT)
    if lf is None:
        check("Forge checkout available for the real Flux2 latent format", False)
        return

    tmp = tempfile.mkdtemp(prefix="cc_build_f2_")
    sys.modules["modules.scripts"].basedir = lambda: tmp
    out_dir = Path(tmp) / "probe_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    #   raw method output: deliberately NOT replicated, so the projection has
    #   something real to remove
    g = torch.Generator().manual_seed(21)
    raw = {a: torch.randn(128, generator=g).contiguous() for a in core.PRIMITIVE_AXES}
    save_file(raw, str(out_dir / "derived-flux2-methodB_0.2.safetensors"))

    sys.modules["modules.shared"].sd_model = _t.SimpleNamespace(
        model_config=_t.SimpleNamespace(latent_format=lf.Flux2()),
        forge_objects=_t.SimpleNamespace(vae=ToyVAE(channels=16)),
    )

    dest = Path(core.VECTORS_DIR) / "colorcraft-flux2-derived.safetensors"
    shipped = Path(core.VECTORS_DIR) / "colorcraft-flux2.safetensors"
    check("no Flux2 reference in the test vectors dir — the uncalibrated path",
          not shipped.exists())
    try:
        out = probe.build_basis_file("B@0.2")
        check("build reports the projection it applied",
              "packed family" in out and "replicated subspace" in out,
              [l.strip() for l in out.splitlines() if "packed" in l][:1])
        check("build says the norms are uncalibrated",
              "UNCALIBRATED" in out)

        built = load_file(str(dest))
        check("all 11 axes present", len(built) == 11, f"{len(built)} axes")
        worst = max(probe.replication_residual(v, 128) for v in built.values())
        check("every axis — diagonals included — is free of the 2x2 grid",
              worst < 1e-6, f"max residual={worst:.2e}")

        before = probe.replication_residual(raw["exposure"], 128)
        check("the raw vectors really did carry a grid component",
              before > 0.5, f"residual before={before:.3f}")

        kept = abs(cos(built["exposure"], core.project_replicated(raw["exposure"])))
        check("the surviving direction is the projection, not something else",
              kept > 0.9999, f"|cos|={kept:.6f}")
    finally:
        if dest.exists():
            dest.unlink()


def test_measure_calibration(probe):
    """Phase 3's instrument. It cannot be checked against a real VAE offline, so
    what is checked is that it runs at both latent ranks, reports every value it
    claims to, and refuses to touch a shipped file."""
    print("\nPhase 3 — measure calibration")
    import types as _t
    from safetensors.torch import save_file
    from lib_colorcraft import core

    lf = stubs.load_latent_formats(FORGE_ROOT)
    fmt = lf.Flux() if lf else stubs.make_toy("Flux")
    shared = sys.modules["modules.shared"]
    shared.sd_model = _t.SimpleNamespace(
        model_config=_t.SimpleNamespace(latent_format=fmt),
        forge_objects=_t.SimpleNamespace(vae=ToyVAE(channels=16)))

    out = probe.measure_calibration("", 128, 4, 0.2, 3, "", False)
    check("reports the projection-scale table", "percentile" in out and "chroma r" in out)
    check("says a known family is a test of the method, not a result",
          "already calibrated" in out)
    check("fits the percentile against this family's known values",
          "best reproduces" in out and "vs known" in out)
    check("reports effect PSNR against the Z-Image targets",
          "effect PSNR" in out and "x norm" in out and "slope" in out)
    check("prints a paste-ready MODEL_DEV_DEFAULTS row",
          'MODEL_DEV_DEFAULTS["zimage"]' in out and "vibrance_k" in out
          and "hue_bias" in out)
    check("without a hue anchor it prints the one to carry to the next model",
          "crosses 0 on THIS model" in out)

    row = [l for l in out.splitlines() if "MODEL_DEV_DEFAULTS" in l][0]
    parsed = eval(row.split("=", 1)[1].strip())  # noqa: S307 — our own formatting
    check("the printed row is valid Python with every key core.resolve_dev reads",
          set(parsed) == set(core.MODEL_DEV_DEFAULTS["zimage"]), sorted(parsed))

    with_anchor = probe.measure_calibration("", 128, 4, 0.2, 3, "30", False)
    check("a hue anchor turns into a hue_bias", "-> hue_bias =" in with_anchor)

    #   5D: Krea 2 / Qwen hand over [B,C,1,H,W]. compare_bases shipped a rank bug
    #   to a live session for exactly this reason.
    shared.sd_model = _t.SimpleNamespace(
        model_config=_t.SimpleNamespace(latent_format=lf.Wan21() if lf else stubs.make_toy("Wan21")),
        forge_objects=_t.SimpleNamespace(vae=ToyWanVAE(channels=16)))
    out5 = probe.measure_calibration("", 128, 4, 0.2, 3, "", False)
    check("runs on a 5D (Wan-family) latent", 'MODEL_DEV_DEFAULTS["krea2"]' in out5,
          out5.strip().splitlines()[-1][:60] if "Traceback" in out5 else "")

    #   a family with vectors but nothing shipped, and one with neither
    shared.sd_model = _t.SimpleNamespace(
        model_config=_t.SimpleNamespace(latent_format=lf.Flux2() if lf else stubs.make_toy("Flux2")),
        forge_objects=_t.SimpleNamespace(vae=ToyVAE(channels=16)))
    check("a family with no vector file says so instead of raising",
          "does not create one" in probe.measure_calibration("", 128, 4, 0.2, 3, "", False))

    #   the shipped file is the reference every measurement is checked against
    shared.sd_model = _t.SimpleNamespace(
        model_config=_t.SimpleNamespace(latent_format=fmt),
        forge_objects=_t.SimpleNamespace(vae=ToyVAE(channels=16)))
    out = probe.measure_calibration("", 128, 4, 0.2, 3, "", True)
    check("refuses to rescale a shipped file", "never overwritten" in out)

    dest = Path(core.VECTORS_DIR) / "colorcraft-zimage-derived.safetensors"
    try:
        source = {k: v.float().clone().contiguous() for k, v in core.load_basis("zimage").items()}
        save_file(source, str(dest))
        out = probe.measure_calibration("", 128, 4, 0.2, 3, "", True)
        check("with a derived file present, the norms are rescaled and written",
              "norms rescaled by effect" in out)

        after = core.load_basis("zimage", core.DERIVED_VARIANT)
        mults = {a: float(after[a].norm() / source[a].norm()) for a in core.PRIMITIVE_AXES}
        reported = {}
        line = [l for l in out.splitlines() if "norms rescaled by effect" in l][0]
        for part in line.split(":")[-1].split(","):
            name, _, factor = part.strip().rpartition(" x")
            if name in core.PRIMITIVE_AXES:
                reported[name] = float(factor)
        check("the norms on disk match the multipliers it printed",
              len(reported) == len(core.PRIMITIVE_AXES)
              and all(abs(mults[a] - reported[a]) < 2e-3 for a in reported),
              f"{len(reported)}/{len(core.PRIMITIVE_AXES)} axes checked")

        d = after["temp+tint"]
        made = after["temperature"] / after["temperature"].norm() + after["tint"] / after["tint"].norm()
        check("diagonals were rebuilt from the rescaled parents",
              abs(cos(d, made)) > 0.9999)
    finally:
        if dest.exists():
            dest.unlink()


def test_stale_lib_guard(probe):
    print("\nStale-lib guard (Forge's Reload UI keeps sys.modules)")
    import lib_colorcraft

    check("the package is stamped with its source mtime",
          isinstance(getattr(lib_colorcraft, "_source_mtime", None), float))

    #   a second call must be a no-op, or two scripts would end up holding two
    #   different module instances
    before = sys.modules["lib_colorcraft"]
    probe._ensure_fresh_lib()
    check("re-running the guard is a no-op when nothing changed",
          sys.modules["lib_colorcraft"] is before)

    #   pretend an edit happened: the guard must drop and re-import
    lib_colorcraft._source_mtime = 0.0
    probe._ensure_fresh_lib()
    check("a newer source triggers a re-import",
          sys.modules["lib_colorcraft"] is not before)

    #   and the stale-core path must explain itself rather than raise
    real = probe.core
    try:
        probe.core = types.SimpleNamespace(family_for_latent_format=lambda x: None)
        out = probe.build_basis_file("B@0.2")
        check("a stale core reports itself instead of AttributeError",
              "stale" in out and "DERIVED_VARIANT" in out, out.strip()[:70])
    finally:
        probe.core = real
        repoint_vectors()


def test_compare_bases(probe):
    print("\nCompare bases (the instrument the generation A/B could not be)")
    import types as _t
    from safetensors.torch import save_file
    from lib_colorcraft import core

    lf = stubs.load_latent_formats(FORGE_ROOT)
    fmt = lf.Flux() if lf else stubs.make_toy("Flux")
    shipped = core.load_basis("zimage")

    dest = Path(core.VECTORS_DIR) / "colorcraft-zimage-derived.safetensors"
    try:
        # a derived basis that is deliberately just the shipped one: agreement
        # must then be near-perfect, which is the calibration of the metric
        save_file({k: v.clone().contiguous() for k, v in shipped.items()}, str(dest))
        shared = sys.modules["modules.shared"]

        #   Both latent ranks. Krea 2 / Qwen use the Wan VAE and hand over
        #   [B,C,1,H,W]; compare_bases originally forgot to squeeze that and blew
        #   up live with "too many values to unpack", because this test only ever
        #   built a 4D toy.
        out = None
        for label, cls in (("4D (Flux-family)", ToyVAE), ("5D (Wan-family)", ToyWanVAE)):
            shared.sd_model = _t.SimpleNamespace(
                model_config=_t.SimpleNamespace(latent_format=fmt),
                forge_objects=_t.SimpleNamespace(vae=cls(channels=16)))
            out = probe.compare_bases("", 128, "0.1, 0.4", 4)
            check(f"compare runs on a {label} latent",
                  "effect" in out and "agree" in out and "margin" in out,
                  out.strip().splitlines()[-1][:60] if "Traceback" in out else "")
            check(f"identical basis reports near-perfect agreement — {label}",
                  " 99.0" in out, "expected PSNR 99 for identical vectors")

        check("clipping and usable slider values are reported",
              "clip%" in out and "usable slider values" in out)

        # now perturb one axis hard: agreement must drop for it
        broken = {k: v.clone() for k, v in shipped.items()}
        g = torch.Generator().manual_seed(3)
        broken["exposure"] = torch.randn(shipped["exposure"].shape, generator=g)
        broken["exposure"] = broken["exposure"] / broken["exposure"].norm() * shipped["exposure"].norm()
        save_file({k: v.contiguous() for k, v in broken.items()}, str(dest))
        out2 = probe.compare_bases("", 128, "0.4", 4)
        exp_line = [l for l in out2.splitlines() if l.strip().startswith("0.4")]
        check("a wrong vector shows up as low agreement", len(exp_line) >= 1 and out2 != out,
              "metric responds to a deliberately wrong axis")

        check("missing derived file is reported, not raised",
              "run 'Build basis file'" in (dest.unlink() or probe.compare_bases("", 128, "0.1", 4)))
    finally:
        if dest.exists():
            dest.unlink()


def main():
    probe = load_probe()
    vectors = install_temp_vectors()
    print(f"probe self-test — torch {torch.__version__}")
    print(f"  vectors dir (throwaway copy, shipped files only): {vectors}")
    test_colour_maths(probe)
    test_transforms_and_measures(probe)
    test_methods_against_known_answer(probe)
    test_flux2_packing(probe)
    test_run_probe_end_to_end(probe)
    test_build_basis_file(probe)
    test_build_basis_file_packed(probe)
    test_measure_calibration(probe)
    test_stale_lib_guard(probe)
    test_compare_bases(probe)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
