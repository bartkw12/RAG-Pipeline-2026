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

import html as html_module
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc.document import DoclingDocument, NodeItem, PictureItem

logger = logging.getLogger(__name__)


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
    """Activate Azure GPT-4.1 vision analysis for figures and
    diagrams detected by the standard pipeline."""

    # ── Azure OpenAI credentials (used when vlm_enabled=True) ──
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

    strip_boilerplate_blocks: bool = True
    """Remove repeated boilerplate content that Docling fails to
    classify as page headers — e.g. security classification banners
    (``THALES GROUP INTERNAL``) and repeated document-ID tables."""

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


# ── Converter builder ───────────────────────────────────────────

# System prompt for GPT-4.1 VLM figure descriptions.
_VLM_SYSTEM_PROMPT = (
    "You are analyzing a technical engineering document. "
    "Describe the content of this figure, diagram, schematic, or chart in detail "
    "(3-5 sentences). Identify: what type of visualization it is (e.g., block diagram, "
    "circuit schematic, waveform, flowchart, test setup photo); the key components, "
    "signals, or data shown; and what technical information or relationships it conveys. "
    "Use precise engineering terminology. Do not describe visual styling, colors, or layout."
)


def _build_converter(config: ParserConfig) -> DocumentConverter:
    """Construct a Docling ``DocumentConverter`` configured per *config*.

    Uses the standard PDF pipeline for layout analysis, OCR, and table
    extraction.  Picture descriptions are handled separately by
    ``_describe_pictures()`` after conversion.
    """

    return _build_standard_converter(config)


def _build_standard_converter(config: ParserConfig) -> DocumentConverter:
    """Build a converter using the standard PDF pipeline.

    Picture classification is enabled so the filter can distinguish
    logos/icons from content figures.  VLM descriptions are done in
    a separate post-conversion step.
    """

    # ── Table structure ─────────────────────────────────────────
    table_mode = (
        TableFormerMode.ACCURATE
        if config.table_mode == "accurate"
        else TableFormerMode.FAST
    )
    table_options = TableStructureOptions(mode=table_mode)

    # ── OCR ─────────────────────────────────────────────────────
    ocr_options: OcrAutoOptions | EasyOcrOptions = (
        EasyOcrOptions(force_full_page_ocr=False)
        if config.ocr_enabled
        else OcrAutoOptions()
    )

    # ── Assemble PDF pipeline options ───────────────────────────
    # Note: do_picture_description=False — we handle VLM separately
    # to avoid Pillow crashes and to inject figure caption context.
    pdf_pipeline_opts = PdfPipelineOptions(
        do_ocr=config.ocr_enabled,
        do_table_structure=True,
        table_structure_options=table_options,
        ocr_options=ocr_options,
        do_picture_classification=True,
        do_picture_description=False,
        generate_picture_images=False,
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_opts,
                pipeline_cls=StandardPdfPipeline,
            ),
            InputFormat.DOCX: WordFormatOption(),
        },
    )


# ── VLM figure description (Azure GPT-4.1) ─────────────────────


def _resolve_ref(doc: DoclingDocument, ref) -> object | None:
    """Resolve a ``RefItem`` (JSON pointer like ``#/texts/42``) on *doc*."""
    cref = getattr(ref, "cref", None)
    if not cref or not isinstance(cref, str):
        return None
    parts = cref.lstrip("#/").split("/")
    if len(parts) != 2:
        return None
    collection_name, index_str = parts
    collection = getattr(doc, collection_name, None)
    if collection is None or not isinstance(collection, list):
        return None
    try:
        return collection[int(index_str)]
    except (ValueError, IndexError):
        return None


def _get_caption_text(doc: DoclingDocument, item: PictureItem) -> str:
    """Extract the caption/title text for a ``PictureItem``."""
    parts: list[str] = []
    for ref in item.captions or []:
        ref_item = _resolve_ref(doc, ref)
        text = getattr(ref_item, "text", "") or ""
        if text.strip():
            parts.append(text.strip())
    return " ".join(parts)


_RE_FIGURE_NUMBER = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)

_RE_FIGURE_PREFIX = re.compile(
    r"^(?:Figure|Fig\.?)\s*\d+\s*[:.\-]\s*", re.IGNORECASE
)


def _extract_figure_number(caption: str) -> str | None:
    """Extract figure number from a caption like 'Figure 15: ...'."""
    m = _RE_FIGURE_NUMBER.search(caption)
    return m.group(1) if m else None


def _caption_without_prefix(caption: str) -> str:
    """Strip the 'Figure N:' prefix from a caption if present."""
    return _RE_FIGURE_PREFIX.sub("", caption).strip()


def _call_azure_vlm(
    image_bytes: bytes,
    prompt: str,
    config: ParserConfig,
) -> str | None:
    """Send an image to Azure GPT-4.1 and return the description text."""
    import base64

    import requests

    base = (config.azure_endpoint or "").rstrip("/")
    url = (
        f"{base}/openai/deployments/{config.azure_model}"
        f"/chat/completions?api-version={config.azure_api_version}"
    )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    try:
        resp = requests.post(
            url,
            headers={"api-key": config.azure_api_key},
            json=payload,
            timeout=60,
        )
        if not resp.ok:
            logger.warning(
                "Azure VLM API returned %s: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Azure VLM API call failed.")
        return None


def _crop_picture_image(
    item: PictureItem,
    pdf_pages: dict[int, Image.Image],
) -> bytes | None:
    """Crop a picture from a pre-rendered page image and return PNG bytes.

    *pdf_pages* maps 1-based page numbers to rendered PIL images.
    Returns None on any error (corrupt crop, missing page, etc.).
    """
    from io import BytesIO

    try:
        if not item.prov:
            return None

        prov = item.prov[0]
        page_no = prov.page_no
        page_img = pdf_pages.get(page_no)
        if page_img is None:
            return None

        bbox = prov.bbox
        # Docling bbox: l, t, r, b in points (72 dpi).
        # Page rendered at scale=2.0 → 144 dpi.
        scale = 2.0
        left = int(bbox.l * scale)
        top = int(bbox.t * scale)
        right = int(bbox.r * scale)
        bottom = int(bbox.b * scale)

        # Clamp to page bounds
        pw, ph = page_img.size
        left = max(0, min(left, pw))
        top = max(0, min(top, ph))
        right = max(left + 1, min(right, pw))
        bottom = max(top + 1, min(bottom, ph))

        cropped = page_img.crop((left, top, right, bottom))
        buf = BytesIO()
        cropped.save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        logger.debug(
            "Failed to crop picture at page %s: %s",
            getattr(item.prov[0], "page_no", "?") if item.prov else "?",
            item.self_ref,
            exc_info=True,
        )
        return None


def _render_pdf_pages(
    pdf_path: Path,
    page_numbers: set[int],
    scale: int = 2,
) -> dict[int, Image.Image]:
    """Render specific PDF pages as PIL images using pypdfium2.

    *page_numbers* are 1-based.  Returns a dict mapping page number
    to its rendered PIL image.
    """
    import pypdfium2

    pages: dict[int, Image.Image] = {}
    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        for page_no in sorted(page_numbers):
            idx = page_no - 1  # pypdfium2 uses 0-based indexing
            if 0 <= idx < len(pdf):
                try:
                    page = pdf[idx]
                    bitmap = page.render(scale=scale)
                    pages[page_no] = bitmap.to_pil()
                except Exception:
                    logger.debug(
                        "Failed to render page %d from '%s'.",
                        page_no, pdf_path.name, exc_info=True,
                    )
        pdf.close()
    except Exception:
        logger.warning(
            "Could not open PDF '%s' for image rendering.", pdf_path.name,
            exc_info=True,
        )
    return pages


def _describe_pictures(
    doc: DoclingDocument,
    pdf_path: Path,
    config: ParserConfig,
) -> dict[str, str]:
    """Call Azure GPT-4.1 for each content-bearing picture.

    Returns a dict mapping ``item.self_ref`` → formatted description
    string like ``[VLM - Figure 3: Test Set-up] This is a ...``.

    Skips logos/icons and pictures that fail to crop.
    """
    if not (config.vlm_enabled and config.describe_images):
        return {}

    if not config.azure_endpoint or not config.azure_api_key:
        logger.warning(
            "VLM enabled but Azure credentials missing — skipping picture descriptions."
        )
        return {}

    # Collect pictures that need descriptions and their page numbers
    pictures: list[tuple[PictureItem, str, str | None, str]] = []
    page_numbers: set[int] = set()
    fig_counter = 0

    for item, _level in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        # Skip logos/icons
        classes = _get_picture_classes(item)
        if classes & _LOGO_ICON_CLASSES:
            continue

        fig_counter += 1

        caption = _get_caption_text(doc, item)
        if not caption:
            logger.debug("Skipping picture %d — no caption.", fig_counter)
            continue

        fig_num = _extract_figure_number(caption)
        fig_label = f"Figure {fig_num}" if fig_num else f"Figure {fig_counter}"

        if item.prov:
            page_numbers.add(item.prov[0].page_no)
        pictures.append((item, caption, fig_num, fig_label))

    if not pictures:
        return {}

    # Render only the pages that contain pictures
    logger.info(
        "Rendering %d page(s) for %d picture(s) ...",
        len(page_numbers), len(pictures),
    )
    pdf_pages = _render_pdf_pages(pdf_path, page_numbers)

    descriptions: dict[str, str] = {}

    for item, caption, fig_num, fig_label in pictures:
        image_bytes = _crop_picture_image(item, pdf_pages)
        if image_bytes is None:
            logger.debug("Skipping %s — could not crop image.", fig_label)
            continue

        # Build a prompt with figure caption context
        prompt_parts = [_VLM_SYSTEM_PROMPT]
        if caption:
            prompt_parts.append(
                f'\nThe document labels this image as: "{caption}".'
            )
        prompt = "\n".join(prompt_parts)

        logger.info("Requesting VLM description for %s ...", fig_label)
        description = _call_azure_vlm(image_bytes, prompt, config)

        if description:
            short_cap = _caption_without_prefix(caption) if caption else ""
            tag = f"[VLM - {fig_label}" + (f": {short_cap}" if short_cap else "") + "]"
            descriptions[str(item.self_ref)] = f"{tag} {description}"
            logger.info("Got VLM description for %s (%d chars).", fig_label, len(description))
        else:
            logger.warning("No VLM description returned for %s.", fig_label)

    return descriptions


# ── Pre-export element filtering ────────────────────────────────

# Labels that correspond to logos, icons, and other non-content images.
_LOGO_ICON_CLASSES: frozenset[str] = frozenset({
    "LOGO", "ICON", "STAMP", "QR_CODE", "BAR_CODE", "SIGNATURE",
})

# ── Boilerplate detection ───────────────────────────────────────

# Regex matching security-classification banner headings that appear
# on every page of Thales-template documents (e.g. "THALES GROUP INTERNAL",
# "THALES GROUP CONFIDENTIAL", "THALES GROUP RESTRICTED", etc.).
_RE_CLASSIFICATION_BANNER = re.compile(
    r"^\s*THALES\s+GROUP\s+\w+\s*$", re.IGNORECASE
)

# Regex matching Entity-ID patterns found in repeated document-ID tables
# (e.g. "15900~9950").
_RE_ENTITY_ID = re.compile(r"\d{4,6}~\d{3,5}")

# Regex matching Thales document identifiers across variants
# (e.g. "7HA-02944-AAAA-980EN", "7HA-02944-ABAA-030EN").
_RE_DOC_ID = re.compile(r"\d+[A-Z]{0,2}-\d{4,6}-[A-Z]{4}-\d{3}[A-Z]{2}")

# Headings that are empty table-of-contents / list-of-X stubs.  The PDF
# parser captures these as section headers, but their body content (the
# actual TOC entries or list items) is never extracted, leaving them as
# contentless noise.
_EMPTY_STUB_HEADINGS: frozenset[str] = frozenset({
    "contents", "table of contents",
    "list of tables", "list of figures",
    "list of abbreviations", "list of acronyms",
    "list of appendices", "list of annexes",
    "list of references", "list of documents",
    "list of drawings", "list of diagrams",
    "list of illustrations", "list of symbols",
})


# Short texts that Docling's layout model sometimes misclassifies as
# section headers.  Common in test-report templates where standalone
# result values ("Passed", "Failed") occupy their own layout region.
_FALSE_POSITIVE_HEADINGS: frozenset[str] = frozenset({
    "passed", "failed", "not applicable", "n/a", "yes", "no",
    "open", "closed", "verified", "not verified", "accepted",
    "rejected", "pending", "cancelled", "deferred", "waived",
    "completed", "incomplete", "ok", "nok", "tbd", "tbc",
})


def _is_false_positive_heading(text: str) -> bool:
    """Return True if *text* is a short non-section phrase that was
    likely misclassified as a section header by the layout model."""
    return text.strip().lower() in _FALSE_POSITIVE_HEADINGS


def _is_classification_banner(text: str) -> bool:
    """Return True if *text* looks like a security-classification heading."""
    return bool(_RE_CLASSIFICATION_BANNER.match(text.strip()))


def _is_boilerplate_table(table_item: object) -> bool:
    """Return True if a ``TableItem`` is a repeated document-ID table.

    Detection heuristic: the table contains a cell matching the
    Entity-ID pattern (``nnnnn~nnnn``) *or* the Thales document-ID
    pattern (``7HA-nnnnn-XXXX-nnnXX``).  These tables appear on every
    page as part of the page header block.
    """
    data = getattr(table_item, "data", None)
    if data is None:
        return False

    for cell in getattr(data, "table_cells", []):
        cell_text = getattr(cell, "text", "")
        if _RE_ENTITY_ID.search(cell_text) or _RE_DOC_ID.search(cell_text):
            return True
    return False


def _filter_document_elements(
    doc: DoclingDocument,
    config: ParserConfig,
    vlm_descriptions: dict[str, str] | None = None,
) -> DoclingDocument:
    """Remove noise elements from a Docling document **in-place**.

    Walks the document tree and collects elements to delete based on
    their ``DocItemLabel`` and (for pictures) their classification
    annotations.  All matching elements are removed in a single
    ``delete_items`` call at the end to avoid iterator invalidation.

    For content-bearing pictures, if ``vlm_descriptions`` contains
    a description (keyed by ``item.self_ref``), the tagged VLM text
    is inserted before deleting the picture.

    Returns the same ``doc`` reference (mutated).
    """
    from docling_core.types.doc.document import (
        DocItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    vlm_map = vlm_descriptions or {}
    items_to_delete: list[NodeItem] = []

    for item, _level in doc.iterate_items():
        if not isinstance(item, DocItem):
            continue

        # ── Headers & footers ───────────────────────────────────
        if config.strip_headers_footers and item.label in (
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
        ):
            items_to_delete.append(item)
            continue

        # ── Table of contents ───────────────────────────────────
        if config.strip_toc and item.label == DocItemLabel.DOCUMENT_INDEX:
            items_to_delete.append(item)
            continue

        # ── Boilerplate: classification banners ─────────────────
        if config.strip_boilerplate_blocks and isinstance(item, SectionHeaderItem):
            if _is_classification_banner(item.text):
                items_to_delete.append(item)
                continue

        # ── Empty stub sections (TOC, List of Figures, etc.) ───
        if config.strip_boilerplate_blocks and isinstance(item, SectionHeaderItem):
            if item.text.strip().lower() in _EMPTY_STUB_HEADINGS:
                items_to_delete.append(item)
                continue

        # ── Boilerplate: repeated document-ID tables ────────────
        if config.strip_boilerplate_blocks and isinstance(item, TableItem):
            if _is_boilerplate_table(item):
                items_to_delete.append(item)
                continue

        # ── Pictures (logos, icons, figures) ────────────────────
        if isinstance(item, PictureItem):
            classification_labels = _get_picture_classes(item)
            is_logo_or_icon = bool(classification_labels & _LOGO_ICON_CLASSES)

            if config.strip_logos_icons and is_logo_or_icon:
                items_to_delete.append(item)
                continue

            # Content-bearing picture — insert VLM description if available.
            ref_key = str(item.self_ref)
            vlm_text = vlm_map.get(ref_key)

            if vlm_text:
                doc.add_text(
                    label=DocItemLabel.PARAGRAPH,
                    text=vlm_text,
                )
                items_to_delete.append(item)
                continue

            # No description — strip picture entirely.
            items_to_delete.append(item)
            continue

    # ── Bulk delete ─────────────────────────────────────────────
    if items_to_delete:
        doc.delete_items(node_items=items_to_delete)
        logger.info(
            "Filtered %d noise element(s) from document.",
            len(items_to_delete),
        )

    return doc


def _get_picture_classes(item: PictureItem) -> set[str]:
    """Extract classification labels from a ``PictureItem``."""
    from docling_core.types.doc.document import PictureClassificationData

    classes: set[str] = set()
    for ann in item.annotations:
        if isinstance(ann, PictureClassificationData):
            for pred in ann.predicted_classes:
                classes.add(pred.class_name.upper())
    return classes





# ── Heading hierarchy inference ─────────────────────────────────

# Regex matching a leading section number like "1", "1.1", "4.2.3", etc.
_RE_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\b")


def _infer_heading_levels(doc: DoclingDocument) -> DoclingDocument:
    """Assign heading depth based on section-numbering patterns.

    Docling's standard PDF pipeline assigns ``level=1`` to every
    ``SectionHeaderItem``, producing flat ``##`` headings.  This
    function infers hierarchy from numbering conventions:

    * ``1``       → level 1 (``##``)
    * ``1.1``     → level 2 (``###``)
    * ``1.5.1``   → level 3 (``####``)
    * ``4.2.3.1`` → level 4 (``#####``)

    Non-numbered headings (e.g. "CONTENTS", "END OF DOCUMENT") keep
    their current level (1).

    Mutates *doc* in-place and returns it.
    """
    from docling_core.types.doc.document import SectionHeaderItem

    adjusted = 0
    for item, _level in doc.iterate_items():
        if not isinstance(item, SectionHeaderItem):
            continue

        m = _RE_SECTION_NUMBER.match(item.text.strip())
        if not m:
            continue

        # Count dot-separated parts: "1" → 1, "1.1" → 2, "4.2.3" → 3
        parts = m.group(1).split(".")
        new_level = len(parts)  # 1-based: 1 → ##, 2 → ###, etc.

        if new_level != item.level:
            item.level = new_level
            adjusted += 1

    if adjusted:
        logger.info("Inferred heading levels for %d section header(s).", adjusted)

    return doc


# ── Post-export Markdown cleanup ─────────────────────────────────

# Regex patterns compiled once at module level.
_RE_EXCESSIVE_BLANKS = re.compile(r"\n{3,}")
_RE_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_RE_PAGE_X_OF_Y = re.compile(
    r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE
)
_RE_REPEATED_SEPARATORS = re.compile(r"(^[ \t]*[-=_]{3,}[ \t]*$\n?){2,}", re.MULTILINE)

# Build a regex that matches headings whose text is a known false-positive
# (e.g. "## Passed", "## Not applicable").  These are test-report result
# values that Docling's layout model misclassified as section headers.
# The regex demotes them to plain text (keeps the value, strips the "## ").
_RE_FALSE_POSITIVE_HEADING = re.compile(
    r"^##\s+(?:" + "|".join(re.escape(h) for h in sorted(_FALSE_POSITIVE_HEADINGS)) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Safety-net regex: catches any residual boilerplate blocks that survived
# the pre-export filter.  Matches a classification banner heading followed
# by a document-ID table (with pipe-delimited rows and separator line).
_RE_RESIDUAL_BOILERPLATE = re.compile(
    r"^#{1,6}\s+THALES\s+GROUP\s+\w+\s*\n"   # "## THALES GROUP INTERNAL"
    r"(?:\s*\n)*"                               # optional blank lines
    r"\|.+\|\s*\n"                              # table header row
    r"\|[-| ]+\|\s*\n"                          # table separator row
    r"(?:\|.+\|\s*\n)*",                        # remaining table rows
    re.IGNORECASE | re.MULTILINE,
)

# Front-matter address block at the start of Thales-template documents.
# Matches a company name heading followed by address, phone, fax, and URL lines.
_RE_FRONT_MATTER = re.compile(
    r"\A\s*"                                    # start of document
    r"(?:#{1,6}\s+.+\n)?"                       # optional heading (company name)
    r"(?:.*\n){0,5}?"                            # up to 5 lines of address
    r"(?=.*(?:Tel\.|Fax\.|www\.))"               # lookahead: block must contain phone/fax/URL
    r"(?:.*\n)*?"                                # consume lines
    r"(?:.*www\.\S+.*\n)",                       # final line with URL
    re.IGNORECASE,
)


def _clean_markdown(md: str) -> str:
    """Post-process a raw Markdown string for chunking quality.

    Applies the following transforms in order:

    1. Strip front-matter address block (company name, phone, fax, URL).
    2. Remove residual "Page X of Y" lines.
    3. Remove residual boilerplate blocks (classification banner + doc-ID table).
    4. Demote false-positive headings ("## Passed" → "Passed").
    5. Collapse repeated separator lines (``---``, ``===``, ``___``).
    6. Decode residual HTML entities (``&amp;`` → ``&``).
    7. Strip HTML comment artifacts (``<!-- ... -->``).
    8. Strip trailing whitespace from every line.
    9. Collapse 3+ consecutive blank lines down to 2.
    10. Strip leading / trailing whitespace from the whole document.
    """
    # 1. Strip front-matter address block
    md = _RE_FRONT_MATTER.sub("", md)

    # 2. Remove "Page X of Y" artifacts
    md = _RE_PAGE_X_OF_Y.sub("", md)

    # 3. Remove residual boilerplate blocks
    md = _RE_RESIDUAL_BOILERPLATE.sub("", md)

    # 4. Demote false-positive headings to plain text
    md = _RE_FALSE_POSITIVE_HEADING.sub(
        lambda m: m.group(0).lstrip("# ").strip(), md
    )

    # 5. Collapse repeated separator lines into a single one
    md = _RE_REPEATED_SEPARATORS.sub("---\n", md)

    # 6. Decode residual HTML entities
    md = html_module.unescape(md)

    # 7. Strip HTML comment artifacts
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)

    # 8. Strip trailing whitespace per line
    md = _RE_TRAILING_WHITESPACE.sub("", md)

    # 9. Collapse excessive blank lines
    md = _RE_EXCESSIVE_BLANKS.sub("\n\n", md)

    # 10. Strip leading/trailing whitespace from the whole document
    md = md.strip()

    return md


# ── Main entry point ────────────────────────────────────────────


def parse_document(
    path: Path,
    doc_id: str,
    config: ParserConfig | None = None,
    converter: DocumentConverter | None = None,
) -> ParsedDocument | None:
    """Parse a single document to clean Markdown.

    Parameters
    ----------
    path:
        Path to the source PDF or DOCX file.
    doc_id:
        SHA-256 hex digest (from the ingestion registry).
    config:
        Parser configuration.  Uses ``ParserConfig()`` defaults if omitted.
    converter:
        Pre-built ``DocumentConverter``.  If omitted, one is built from
        *config*.  Pass a shared converter for batch runs to avoid
        reloading models for every file.

    Returns
    -------
    ParsedDocument
        On success — contains the cleaned Markdown and metadata.
    None
        On failure — the error is logged, caller decides how to handle.
    """
    from docling.datamodel.base_models import ConversionStatus

    from ..config.paths import MARKDOWN_DIR

    cfg = config or ParserConfig()

    # ── 1. Build or reuse converter ─────────────────────────────
    conv = converter or _build_converter(cfg)

    # ── 2. Convert document via Docling ─────────────────────────
    try:
        result = conv.convert(source=path, raises_on_error=False)
    except Exception:
        logger.exception("Docling conversion failed for '%s'.", path.name)
        return None

    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        # Docling may report FAILURE even when most pages were extracted.
        # Proceed if the document object contains any usable content.
        doc = result.document
        has_content = doc is not None and any(True for _ in doc.iterate_items())
        if has_content:
            logger.warning(
                "Conversion returned status '%s' for '%s' — "
                "proceeding with partially extracted content.",
                result.status.name,
                path.name,
            )
        else:
            logger.error(
                "Conversion returned status '%s' for '%s' with no usable content.",
                result.status.name,
                path.name,
            )
            return None
    else:
        doc = result.document

    # ── 3. Extract metadata before filtering ────────────────────
    page_count = result.input.page_count

    # Title: prefer the document's own name, fall back to first heading
    title = doc.name or ""
    if not title:
        from docling_core.types.doc.labels import DocItemLabel

        for item, _ in doc.iterate_items():
            label = getattr(item, "label", None)
            if label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
                title = getattr(item, "text", "") or ""
                if title:
                    break

    if not title:
        title = path.stem

    # ── 3b. VLM figure descriptions (Azure GPT-4.1) ────────────
    vlm_descriptions = _describe_pictures(doc, path, cfg)

    # ── 4. Filter noise elements ────────────────────────────────
    _filter_document_elements(doc, cfg, vlm_descriptions=vlm_descriptions)

    # ── 4b. Infer heading hierarchy from section numbering ──────
    _infer_heading_levels(doc)

    # ── 5. Export to Markdown ───────────────────────────────────
    raw_md = doc.export_to_markdown(
        escape_underscores=False,
        escape_html=False,
    )

    # ── 6. Clean the Markdown ───────────────────────────────────
    cleaned_md = _clean_markdown(raw_md)

    if not cleaned_md:
        logger.warning("Parsing produced empty Markdown for '%s'.", path.name)
        return None

    # ── 7. Write to cache ───────────────────────────────────────
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MARKDOWN_DIR / f"{doc_id}.md"
    out_path.write_text(cleaned_md, encoding="utf-8")
    logger.info(
        "Saved parsed Markdown for '%s' → '%s' (%d chars).",
        path.name,
        out_path.name,
        len(cleaned_md),
    )

    # ── 8. Return result ────────────────────────────────────────
    return ParsedDocument(
        doc_id=doc_id,
        source_path=path,
        markdown=cleaned_md,
        title=title,
        page_count=page_count,
        metadata={
            "source_format": result.input.format.name if result.input.format else "UNKNOWN",
            "conversion_status": result.status.name,
            "file_size_bytes": result.input.filesize,
        },
    )
