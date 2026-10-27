"""Configuration loading and path helpers.

The primary config is ``config.json`` so that the project runs on a bare
standard-library Python. A ``config.yaml`` next to it is also accepted when
PyYAML happens to be installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_JSON = PROJECT_ROOT / "config.json"
DEFAULT_CONFIG_YAML = PROJECT_ROOT / "config.yaml"


def _read(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "PyYAML is not installed. Use config.json, or run "
                "`pip install -r requirements.txt`."
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the config file and attach the resolved project root."""
    if path is not None:
        cfg_path = Path(path)
    else:
        cfg_path = DEFAULT_CONFIG_JSON if DEFAULT_CONFIG_JSON.exists() else DEFAULT_CONFIG_YAML

    cfg = _read(cfg_path)
    cfg["_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path | None = None) -> Path:
    """Persist a config dict back to JSON (used by scripts that mutate paths)."""
    cfg_path = Path(path) if path else Path(cfg.get("_config_path", DEFAULT_CONFIG_JSON))
    payload = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cfg_path


def resolve(cfg: Dict[str, Any], value: str | Path | None) -> Path | None:
    """Resolve a possibly relative path against the project root."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_root"]) / path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
