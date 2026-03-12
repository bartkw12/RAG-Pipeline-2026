from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class EnvironmentError(RuntimeError):
    pass


def load_local_env(env_file: str = ".env", base_dir: str | Path | None = None, override: bool = False) -> None:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    file_path = root / env_file

    if not file_path.exists():
        return

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if not key:
            continue

        if override or key not in os.environ:
            os.environ[key] = value


def require_env(keys: Iterable[str]) -> dict[str, str]:
    missing: list[str] = []
    values: dict[str, str] = {}

    for key in keys:
        value = os.getenv(key)
        if value is None or value == "":
            missing.append(key)
            continue
        values[key] = value

    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise EnvironmentError(f"Missing required environment variables: {missing_sorted}")

    return values


def load_azure_vlm_config(
    config_path: Path | None = None,
    section: str = "OpenAI",
) -> dict[str, str]:
    """Load Azure OpenAI credentials for the VLM pipeline.

    Parameters
    ----------
    config_path:
        Path to the JSON config file.  Defaults to ``CONFIG_FILE``
        from ``paths.py``.
    section:
        Top-level key in the JSON (e.g. ``"OpenAI"`` or ``"OpenAI2"``).

    Returns
    -------
    dict
        Keys: ``endpoint``, ``api_key``, ``model``, ``api_version``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If the requested *section* is missing or lacks required fields.
    """
    import json

    from .paths import CONFIG_FILE

    path = config_path or CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if section not in data:
        raise KeyError(
            f"Section '{section}' not found in config. "
            f"Available: {', '.join(data.keys())}"
        )

    sec = data[section]

    endpoint = sec.get("endpoint") or sec.get("base_url")
    api_key = sec.get("key") or sec.get("api_key")
    api_version = sec.get("api_version") or sec.get("version", "2023-05-15")

    if not endpoint:
        raise KeyError(f"No 'endpoint' or 'base_url' in section '{section}'.")
    if not api_key:
        raise KeyError(f"No 'key' or 'api_key' in section '{section}'.")

    # Pick the first model listed, or fall back to "gpt-4.1"
    models = sec.get("models", {})
    model = sec.get("model") or (next(iter(models.values())) if models else "gpt-4.1")

    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "model": model,
        "api_version": api_version,
    }
