"""Document parser — converts raw PDF/DOCX documents to clean Markdown.

Uses the Docling library for document conversion with optional OCR (EasyOCR)
and VLM (Azure OpenAI or local SmolVLM) support.  The pipeline is:

1. **Convert** — Docling parses the raw document into a structured model.
2. **Filter**  — Remove noise elements (headers, footers, TOC, logos, etc.).
3. **Describe** — If VLM is enabled, generate text descriptions of figures.
4. **Export**  — Convert the cleaned document model to Markdown.
5. **Clean**   — Post-process the Markdown string (collapse whitespace, etc.).
6. **Write**   — Save the final Markdown to ``cache/markdown/{doc_id}.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ── Configuration ───────────────────────────────────────────────


@dataclass(frozen=True)
class ParserConfig:
    """All knobs for the document parsing pipeline.

    Sensible defaults are provided for every field.  Override via CLI
    flags (``--ocr``, ``--vlm``, etc.) or by constructing directly.
    """

    # ── OCR ─────────────────────────────────────────────────────
    ocr_enabled: bool = False
    """Activate EasyOCR for scanned / image-heavy pages."""

    # ── VLM (vision-language model) ─────────────────────────────
    vlm_enabled: bool = False
    """Activate a vision-language model for layout understanding
    and image description."""

    vlm_backend: Literal["azure", "local"] = "azure"
    """Which VLM backend to use.
    - ``"azure"``  — Azure OpenAI endpoint (requires credentials).
    - ``"local"``  — Docling's built-in SmolVLM (runs on-device).
    """

    # ── Azure OpenAI credentials (only used when vlm_backend="azure") ──
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_model: str = "gpt-4.1"
    azure_api_version: str = "2023-05-15"

    # ── Table extraction ────────────────────────────────────────
    table_mode: Literal["accurate", "fast"] = "accurate"
    """Docling ``TableFormerMode``: ``"accurate"`` is slower but better;
    ``"fast"`` is lighter on resources."""

    # ── Post-processing / cleanup toggles ───────────────────────
    strip_headers_footers: bool = True
    """Remove elements classified as page headers or footers."""

    strip_toc: bool = True
    """Remove table-of-contents sections."""

    strip_logos_icons: bool = True
    """Remove images classified as logos or icons."""

    strip_page_numbers: bool = True
    """Remove standalone page-number elements."""

    describe_images: bool = True
    """When VLM is enabled, generate a text description for
    content-bearing figures before stripping the image data.
    Has no effect when ``vlm_enabled`` is False."""


# ── Parser output ───────────────────────────────────────────────


@dataclass(frozen=True)
class ParsedDocument:
    """Result of successfully parsing a single document.

    Attributes
    ----------
    doc_id:
        SHA-256 hex digest of the original file (from the registry).
    source_path:
        Absolute path to the original document that was parsed.
    markdown:
        Cleaned Markdown content ready for chunking.
    title:
        Document title extracted from metadata or the first heading.
    page_count:
        Number of pages in the source document.
    metadata:
        Any additional metadata extracted during conversion
        (author, date, language, etc.).
    """

    doc_id: str
    source_path: Path
    markdown: str
    title: str
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
 