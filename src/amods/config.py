"""
YAML config loading for the concealer/vad/forecaster/stream config "components".

A config is named by component (e.g. "vad") and name (e.g. "default_source", or
"default_source.yaml"). Resolution tries, in order:

1. ``name`` itself, if it looks like a path (contains a path separator) --
   used as-is.
2. ``<search_dir>/<component>/<name>``, if ``search_dir`` is given -- lets one
   config point at a sibling config of a different component next to it (e.g. a
   concealer config naming the VAD config that lives beside it).
3. ``./configs/<component>/<name>``, relative to the current working directory --
   the layout used when running from a checkout of this repository, or any
   directory with a user-provided ``configs/`` folder.
4. ``~/.amods/configs/<component>/<name>`` -- settings persisted by the GUI
   (see :mod:`amods.gui`), so the CLI picks up whatever was last configured
   there regardless of the current working directory.
5. The package's own bundled default configs, so the CLI and GUI keep
   working after a plain ``pip install`` with no ``configs/`` directory
   anywhere in reach.
"""
import os
from importlib import resources
from pathlib import Path

import yaml

_PACKAGE_CONFIGS = "amods.configs"


def user_config_dir() -> Path:
    """Directory where the GUI persists the settings it writes."""
    return Path.home() / ".amods" / "configs"


def _normalize_name(name: str) -> str:
    """Append ``.yaml`` if missing; raise if ``name`` already has a different extension."""
    if not name.endswith(".yaml"):
        if "." in name:
            raise ValueError(f"Invalid file extension for config: {name}. Expected .yaml")
        name += ".yaml"
    return name


def _is_path(name: str) -> bool:
    """True if ``name`` looks like a path (contains a separator) rather than a bare config name."""
    return "/" in name or os.sep in name


def load_yaml_config(path) -> dict:
    """Load a YAML file from an explicit path into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(component: str, name: str, search_dir: str = None) -> dict:
    """Resolve and load a YAML config file for the given ``component``."""
    name = _normalize_name(name)

    if _is_path(name):
        return load_yaml_config(name)

    candidates = []
    if search_dir:
        candidates.append(Path(search_dir) / component / name)
    candidates.append(Path("configs") / component / name)
    candidates.append(user_config_dir() / component / name)

    for path in candidates:
        if path.is_file():
            return load_yaml_config(path)

    packaged = resources.files(_PACKAGE_CONFIGS).joinpath(component, name)
    if packaged.is_file():
        with resources.as_file(packaged) as path:
            return load_yaml_config(path)

    tried = ", ".join(str(c) for c in candidates + [f"<packaged {component}/{name}>"])
    raise FileNotFoundError(f"Could not find {component} config '{name}' (tried: {tried})")


def resolve_config(component: str, name_or_config, search_dir: str = None) -> dict:
    """
    Like :func:`load_config`, but also accepts an already-built config dict
    (used directly, as a defensive copy) instead of a name/path - lets code
    embedding amods (e.g. :class:`amods.stream.Stream`) skip YAML entirely:

        config = load_config("concealer", "default")
        config["denoise"] = False          # override just what you need
        Stream(concealer_config_name=config, ...)
    """
    if isinstance(name_or_config, dict):
        return dict(name_or_config)
    return load_config(component, name_or_config, search_dir=search_dir)
