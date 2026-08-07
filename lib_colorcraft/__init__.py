"""Colorcraft's frontend-agnostic core.

Imported two ways and both must keep working:
  * as `<comfy-package>.lib_colorcraft` from the ComfyUI node (`nodes.py`);
  * as a top-level `lib_colorcraft` from the Forge Neo script, since
    `modules/scripts.py:505` puts every extension's basedir on `sys.path`.

Relative imports inside this package resolve under both, so nothing here may
import it by absolute name.
"""

from . import core, engine, params, spec  # noqa: F401

__all__ = ["core", "engine", "params", "spec"]
