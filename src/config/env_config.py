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
