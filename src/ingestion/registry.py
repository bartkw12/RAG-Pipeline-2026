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

# ── Data classes ────────────────────────────────────────────────


class FileStatus(str, Enum):
    """Result of checking a file against the registry."""

    NEW = "new"                # never seen before
    UNCHANGED = "unchanged"    # same content hash already registered
    MODIFIED = "modified"      # same filename seen before, but content changed


@dataclass
class RegistryEntry:
    """One record in the ingestion registry."""

    doc_id: str                 # SHA-256 hex of file bytes
    filename: str               # original file name (stem + suffix)
    source_path: str            # absolute path at time of ingestion
    size_bytes: int
    ingested_at: str            # ISO-8601 UTC timestamp
    hash: str                   # duplicate of doc_id (explicit for clarity)


@dataclass
class CheckResult:
    """Returned by ``check_file`` so callers know what happened and why."""

    status: FileStatus
    doc_id: str                 # hash of the file being checked
    message: str                # human-readable explanation
    existing_entry: RegistryEntry | None = None   # populated for UNCHANGED / MODIFIED


# ── Registry class ──────────────────────────────────────────────


@dataclass
class IngestionRegistry:
    """In-memory registry backed by a JSON file on disk.

    Typical usage::

        registry = IngestionRegistry.load()

        for path in selected_files:
            result = registry.check_file(path)
            if result.status == FileStatus.UNCHANGED:
                print(result.message)   # "Skipped (unchanged): spec.pdf"
                continue
            # … parse the file …
            registry.register_file(path)

        registry.save()
    """

    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    _path: Path = field(default_factory=lambda: REGISTRY_FILE, repr=False)

    # ── Persistence ─────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path | None = None) -> IngestionRegistry:
        """Load registry from disk.  Returns an empty registry if the file
        doesn't exist yet (first run)."""
        path = path or REGISTRY_FILE
        if not path.exists():
            logger.info("No existing registry found — starting fresh.")
            return cls(_path=path)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Registry file is corrupt or unreadable (%s). "
                           "Starting with an empty registry.", exc)
            return cls(_path=path)

        entries: dict[str, RegistryEntry] = {}
        for doc_id, data in raw.items():
            try:
                entries[doc_id] = RegistryEntry(**data)
            except TypeError:
                logger.warning("Skipping malformed registry entry: %s", doc_id)
        return cls(entries=entries, _path=path)

    def save(self) -> None:
        """Persist the current registry to disk (atomic-ish write)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {doc_id: asdict(entry) for doc_id, entry in self.entries.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self._path)          # atomic on same filesystem
        logger.info("Registry saved (%d entries).", len(self.entries))

    # ── Core operations ─────────────────────────────────────────

    def check_file(self, path: Path) -> CheckResult:
        """Check a file against the registry **without** registering it.

        Returns a ``CheckResult`` with a ``status`` field the caller can
        branch on, plus a human-readable ``message`` suitable for logging
        or displaying to the user.
        """
        file_hash = compute_file_hash(path)

        # Case 1: exact content already ingested
        if file_hash in self.entries:
            existing = self.entries[file_hash]
            return CheckResult(
                status=FileStatus.UNCHANGED,
                doc_id=file_hash,
                message=(
                    f"Skipped (unchanged): '{path.name}' — identical content "
                    f"was already ingested as '{existing.filename}' "
                    f"on {existing.ingested_at}."
                ),
                existing_entry=existing,
            )

        # Case 2: same filename but different hash → content was updated
        for existing in self.entries.values():
            if existing.filename == path.name and existing.hash != file_hash:
                return CheckResult(
                    status=FileStatus.MODIFIED,
                    doc_id=file_hash,
                    message=(
                        f"Re-ingesting (modified): '{path.name}' — content "
                        f"changed since last ingestion on {existing.ingested_at}."
                    ),
                    existing_entry=existing,
                )

        # Case 3: completely new file
        return CheckResult(
            status=FileStatus.NEW,
            doc_id=file_hash,
            message=f"New file: '{path.name}' — queued for ingestion.",
        )

    def register_file(self, path: Path, *, doc_id: str | None = None) -> str:
        """Record a file as successfully ingested.

        If ``doc_id`` is not provided it will be computed (hashed) again.
        Pass the ``doc_id`` from a prior ``check_file`` call to avoid
        double-hashing.

        If the file was previously ingested under a different hash (modified
        content), the old entry is removed automatically.

        Returns the ``doc_id``.
        """
        file_hash = doc_id or compute_file_hash(path)

        # Remove any stale entry for the same filename (handles MODIFIED case)
        stale_ids = [
            did for did, e in self.entries.items()
            if e.filename == path.name and did != file_hash
        ]
        for stale in stale_ids:
            del self.entries[stale]
            logger.debug("Removed stale registry entry %s for '%s'.",
                         stale[:12], path.name)

        entry = RegistryEntry(
            doc_id=file_hash,
            filename=path.name,
            source_path=str(path.resolve()),
            size_bytes=path.stat().st_size,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            hash=file_hash,
        )
        self.entries[file_hash] = entry
        logger.info("Registered '%s' (hash %s…).", path.name, file_hash[:12])
        return file_hash

    # ── Convenience ─────────────────────────────────────────────

    def is_already_ingested(self, path: Path) -> bool:
        """Quick boolean check — True if the file's content hash is in the
        registry."""
        return compute_file_hash(path) in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.entries


