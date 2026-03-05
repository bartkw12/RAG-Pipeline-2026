"""Ingestion registry — tracks which files have been ingested.

Every ingested file is identified by a **doc_id** which is the SHA-256 hash of
its raw bytes.  This means:

* Same content → same doc_id, regardless of filename or location.
* If a file is renamed but unchanged → still recognised as already ingested.
* If a file keeps its name but content changes → new hash → treated as an
  update and re-ingested.

The registry is persisted as a single JSON file at ``REGISTRY_FILE``
(``cache/meta/ingestion_registry.json`` by default).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from config.paths import REGISTRY_FILE

logger = logging.getLogger(__name__)

# ── Public helpers ──────────────────────────────────────────────

# How many bytes to read at a time when hashing large files (8 MB).
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Reads the file in chunks so large files don't blow up memory.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            sha.update(chunk)
    return sha.hexdigest()



