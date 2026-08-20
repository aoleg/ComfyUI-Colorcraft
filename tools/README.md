# tools/

Development tools. Nothing in here runs as part of Colorcraft — Forge only loads
`scripts/`, so these files are inert until you deliberately put one there.

## colorcraft_probe.py — deriving a basis for a new VAE family

Colorcraft grades along a set of colour axes stored in `vectors/`, one file per
VAE family. This tool is how those files get made for a family that doesn't have
one yet: it works out the axes from the VAE itself, measures how strong each one
should be, and writes the file.

To use it, copy it into `scripts/` and reload the Forge UI; a
**"Colorcraft — vector probe (dev)"** panel appears on the txt2img tab. Delete it
from `scripts/` again when you're done, so ordinary users never see it.

```
cp tools/colorcraft_probe.py scripts/
```

The four buttons, in the order you'd use them:

1. **Run probe** — loads a folder of your own photographs, applies a known edit to
   each one (brighter, warmer, greener…), and watches what that does inside the
   VAE. The result is a candidate set of axes.
2. **Build basis file** — turns one of those candidate sets into a real vectors
   file, fixing up signs and strengths and filling in the four diagonal axes.
3. **Compare bases** — only useful when the family already has a file to compare
   against. It applies both sets to the same image and reports whether they do
   the same thing, which a side-by-side generation cannot reliably show.
4. **Measure calibration** — measures the handful of per-family numbers that
   masking and vibrance depend on, and prints them ready to paste into
   `lib_colorcraft/core.py`.

Run steps 1 and 4 on **Krea 2 and Z-Image first**. Those two already have known
correct values, so the report tells you how close the measurement lands — and a
method that can't reproduce a known answer shouldn't be trusted on an unknown
one.

`tests/probe_selftest.py` checks everything in here that can be checked without a
GPU, and runs from the repo root regardless of where this file sits.
